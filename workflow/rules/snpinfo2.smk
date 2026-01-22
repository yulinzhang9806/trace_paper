#!python3

import numpy as np
import pandas as pd
import math
from workflow.scripts.utils import SNPINFO

rule snpinfo:
    """Get SNP info for all biallelic snps in HGDP."""
    input:
        hgdp_bcf = "results/realdata/{dataset}/vcf_files/hgdp_1kg_chr{chrom}.bcf",
        ancestral =  paths['human_ancestor'] + "{chrom}.fa",
    params:
        outpref = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.chr{chrom}",
    output:
        txtfile = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.chr{chrom}.txt",
    run:
        SNPINFO().get_snp_info(input.hgdp_bcf, input.ancestral, params.outpref)

rule snpinfo_archaic:
    """Get SNP info for archaics given SNPs."""
    input:
        txtfile = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.chr{chrom}.txt",
        archaic = paths['archaic_bcf'] + ".{chrom}.split_multiallelic.bcf",
    params:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.chr{chrom}",
        archaicbcf = paths['archaic_bcf'] + ".{chrom}.split_multiallelic",
        outpref = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.chr{chrom}",
    output:
        txtfile="results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.chr{chrom}.txt",
    run:
        SNPINFO().append_archaic_snpinfo(params.snpinfo, params.archaicbcf, params.outpref)

rule snpinfo_afr:
    """Get SNP info for African DAF given SNPs."""
    input:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.chr{chrom}.txt",
        hgdp_bcf = "results/realdata/{dataset}/vcf_files/hgdp_1kg_chr{chrom}.bcf",
        afr_poplabel = "results/realdata/{dataset}/AFR.sample",
        gbr_poplabel = "results/realdata/{dataset}/AFR.sample",
        yri_poplabel = "results/realdata/{dataset}/AFR.sample",
        yri_gbr_poplabel = "results/realdata/{dataset}/AFR.sample",
    params:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.chr{chrom}",
        outpref = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.chr{chrom}",
    output:
        txtfile="results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.chr{chrom}.txt",
    run:
        SNPINFO().append_AFR_info(params.snpinfo, input.hgdp_bcf, input.gbr_poplabel, "AltAF_GBR", params.outpref)
        SNPINFO().append_AFR_info(params.outpref, input.hgdp_bcf, input.yri_poplabel, "AltAF_YRI", params.outpref)
        SNPINFO().append_AFR_info(params.outpref, input.hgdp_bcf, input.yri_gbr_poplabel, "AltAF_YRI_GBR", params.outpref)
        SNPINFO().append_AFR_info(params.outpref, input.hgdp_bcf, input.afr_poplabel, "AltAF_AFR", params.outpref)

rule snpinfo_outgroup:
    """Get SNP info for African DAF given SNPs."""
    input:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.chr{chrom}.txt",
        outgrouopfile = "results/realdata/{dataset}/hmmix/outgroup_variants.txt"
    params:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.chr{chrom}",
        outpref = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.chr{chrom}",
    output:
        txtfile="results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.chr{chrom}.txt",
    run:
        SNPINFO().append_outgroup_info(params.snpinfo, input.outgrouopfile, params.outpref)

rule snpinfo_strictmask:
    """Get SNP info for if in strictmask."""
    input:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.chr{chrom}.txt",
        strictmask = "results/realdata/{dataset}/hmmix/hg38_strictmask.bed",
        bcffile = "results/realdata/{dataset}/vcf_files/hgdp_1kg_chr{chrom}.bcf",
    params:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.chr{chrom}",
        outpref = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.chr{chrom}",
    output:
        txtfile="results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.chr{chrom}.txt",
    run:
        SNPINFO().append_strictmask_info(params.snpinfo, input.strictmask, input.bcffile, params.outpref)

rule snpinfo_manifesto:
    """Get SNP info for if in manifesto."""
    input:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.chr{chrom}.txt",
        manifesto = "/global/scratch/p2p3/pl1_moorjani/SHARED_LAB/DATASETS/hg38/ANCIENT/ARCHAIC/manifesto_hg38/chr{chrom}.bed",
        bcffile = "results/realdata/{dataset}/vcf_files/hgdp_1kg_chr{chrom}.bcf",
    params:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.chr{chrom}",
        outpref = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.manifesto.chr{chrom}",
    output:
        txtfile="results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.manifesto.chr{chrom}.txt",
    run:
        SNPINFO().append_manifesto_info(params.snpinfo, input.manifesto, input.bcffile, params.outpref)

rule snpinfo_mutage:
    """Get SNP info for mutage."""
    input:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.chr{chrom}.txt",
    params:
        snpinfo = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.chr{chrom}",
        mutage_pref = "results/realdata/{dataset}/singer/mutage_files/mutage_chr{chrom}_",
        mutage_range = range(250, 300),
        outpref = "results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.mutage.chr{chrom}",
    output:
        txtfile="results/realdata/{dataset}/snpinfo/hgdp_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.mutage.chr{chrom}.txt",
    run:
        SNPINFO().append_mutage_info(params.snpinfo, params.mutage_pref, params.mutage_range, params.outpref)