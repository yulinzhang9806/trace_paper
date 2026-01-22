#!python3
import numpy as np
import pandas as pd
import math
import tskit
import tszip
import json

# rule preprocess_1000g_vcf:
#     """Process 1000 Genomes VCF raw data, exclude missing, include only biallelic SNPs, apply pilot mask, subset to YRI, GBR"""
#     input:
#         vcf = paths['1000g_raw_vcf'] + "{chrom}" + paths['1000g_raw_vcf_suffix'],
#         mask = paths['pilot_mask_hg38'],
#         sample = "results/YRI_GBR.sample",
#     output:
#         vcf = "results/realdata/1000g/input_vcf/1000g_hg38_chr{chrom}.vcf.gz",
#     shell:
#         """
#         vcftools --gzvcf {input.vcf} --keep {input.sample} --bed {input.mask} --remove-indels --max-missing 1.0 \
#          --min-alleles 2 --max-alleles 2 --recode --recode-INFO-all --stdout | bgzip -c > {output.vcf}
#         """


# ---------- Relate ------------
rule relate_subset:
    """Subset to 1000 Genomes YRI and GBR, reestimate branch length."""
    input:
        anc="results/realdata/{dataset}/relate/relate_all_samples/chr{chrom}_hg38_human_ancestor.anc",
        mut="results/realdata/{dataset}/relate/relate_all_samples/chr{chrom}_hg38_human_ancestor.mut",
        poplabels="results/realdata/{dataset}/relate/relate_all_samples/all.poplabel",
        relate=paths["relate"],
    params:
        outpref="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR",
    output:
        subset_anc="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR.anc",
        subset_mut="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR.mut",
        subset_poplabels="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR.poplabels",
    shell:
        """
        {input.relate}/bin/RelateExtract --mode SubTreesForSubpopulation \
        --anc {input.anc} \
        --mut {input.mut} \
        --poplabels {input.poplabels} \
        --pop_of_interest YRI,GBR \
        -o {params.outpref}
        """


rule relate_reestimate:
    """Reestimate branch length for 1000 Genomes YRI and GBR."""
    input:
        anc="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR.anc",
        mut="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR.mut",
        poplabels="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR.poplabels",
        relate=paths["relate"],
    params:
        inpref="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR",
        mutrate=1.25e-8,
        seed=10,
        outpref="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR_popsize",
    output:
        reestimate_anc="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR_popsize.anc.gz",
        reestimate_mut="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR_popsize.mut.gz",
    threads: 32
    shell:
        """
        {input.relate}/scripts/EstimatePopulationSize/EstimatePopulationSize.sh \
        -i {params.inpref} \
        -m {params.mutrate} \
        --threads {threads} \
        --seed {params.seed} \
        --poplabels {input.poplabels} \
        -o {params.outpref}
        """


rule relate_to_ts:
    """Transfer Relate output to ts trees."""
    input:
        anc="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR_popsize.anc.gz",
        mut="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_hg38_human_ancestor_YRI_GBR_popsize.mut.gz",
        relate_ts=str(paths["relate_lib"] + "/bin/Convert"),
    params:
        workingdir="results/realdata/{dataset}/relate/relate_YRI_GBR/",
        inpref="chr{chrom}_hg38_human_ancestor_YRI_GBR_popsize",
        outpref="chr{chrom}_YRI_GBR",
    output:
        ts="results/realdata/{dataset}/relate/relate_YRI_GBR/chr{chrom}_YRI_GBR.trees",
    shell:
        """
        cd {params.workingdir}
        {input.relate_ts} --mode ConvertToTreeSequence --anc {params.inpref}.anc --mut {params.inpref}.mut -o {params.outpref}
        """


# ----------- hmmix --------------
rule get_bcf:
    input:
        phased_vcf="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}.vcf.gz",
    output:
        keep_bcf="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}.bcf",
        keep_bcf_index="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}.bcf.csi",
    shell:
        """
        bcftools view -v snps {input.phased_vcf} -l 1 -O b > {output.keep_bcf}
        bcftools index {output.keep_bcf}
        """


rule create_outgroup:
    input:
        keep_vcf=expand(
            "results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom_id}.bcf",
            chrom_id=range(1, 23),
            allow_missing=True,
        ),
        ind_json="results/realdata/{dataset}/hmmix/individual.json",
        weights="results/realdata/{dataset}/hmmix/hg38_strictmask.bed",
    output:
        outgroup="results/realdata/{dataset}/hmmix/outgroup_variants.txt",
        mutationrate="results/realdata/{dataset}/hmmix/mutationrate.bed",
    params:
        bcfpref="results/realdata/{dataset}/vcf_files/1000g_hg38_chr",
        ancestral=paths["human_ancestor"],
        reference=paths["human_reference_hg38"],
        window_size=1000000,
    shell:
        """
        hmmix create_outgroup -ind={input.ind_json} -vcf={params.bcfpref}*.bcf -weights={input.weights} -out={output.outgroup} -ancestral={params.ancestral}*.fa -refgenome={params.reference}*.fa
        hmmix mutation_rate -outgroup={output.outgroup} -weights={input.weights} -window_size={params.window_size} -out={output.mutationrate}
        """


rule create_ingroup:
    input:
        keep_vcf=expand(
            "results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom_id}.bcf",
            chrom_id=range(1, 23),
            allow_missing=True,
        ),
        weights="results/realdata/{dataset}/hmmix/hg38_strictmask.bed",
        outgroup="results/realdata/{dataset}/hmmix/outgroup_variants.txt",
    output:
        obs="results/realdata/{dataset}/hmmix/00obs/obs.{ind}.txt",
    params:
        bcfpref="results/realdata/{dataset}/vcf_files/1000g_hg38_chr",
        ancestral=paths["human_ancestor"],
        obspref="results/realdata/{dataset}/hmmix/00obs/obs",
        ind=lambda wildcards: wildcards.ind,
    shell:
        """
        hmmix create_ingroup -ind={params.ind} -vcf={params.bcfpref}*.bcf -weights={input.weights} -out={params.obspref} -ancestral={params.ancestral}*.fa -outgroup={input.outgroup}
        """


rule train_decode_hmmix:
    input:
        outgroup="results/realdata/{dataset}/hmmix/outgroup_variants.txt",
        mutationrate="results/realdata/{dataset}/hmmix/mutationrate.bed",
        ind_json="results/realdata/{dataset}/hmmix/individual.json",
        weights="results/realdata/{dataset}/hmmix/hg38_strictmask.bed",
        obs="results/realdata/{dataset}/hmmix/00obs/obs.{ind}.txt",
        archaic_bcf=expand(
            "/global/scratch/p2p3/pl1_moorjani/lauritsskov2/ArchaicSegments/helperfiles/archaic_variants/hg38/individuals.{chrom_id}.bcf",
            chrom_id=range(1, 23),
        ),
    params:
        individual=lambda wildcards: wildcards.ind,
        hmmix_out="results/realdata/{dataset}/hmmix/02decode/{ind}",
        archaic_bcf="/global/scratch/p2p3/pl1_moorjani/lauritsskov2/ArchaicSegments/helperfiles/archaic_variants/hg38/individuals.",
    output:
        hmmix_trained="results/realdata/{dataset}/hmmix/01train_haploid/trained.{ind}.json",
        hmmix_out1="results/realdata/{dataset}/hmmix/02decode/{ind}.hap1.txt",
        hmmix_out2="results/realdata/{dataset}/hmmix/02decode/{ind}.hap2.txt",
    shell:
        """
        hmmix train -obs={input.obs} -weights={input.weights} -mutrates={input.mutationrate} -out={output.hmmix_trained} -haploid
        hmmix decode -obs={input.obs} -weights={input.weights} -mutrates={input.mutationrate} -param={output.hmmix_trained} -admixpop={params.archaic_bcf}*.bcf -haploid -out={params.hmmix_out}
        """


# ------------------------ Run SINGER -------------------------- #


rule subset_vcf:
    """Subset VCF to 1000 Genomes YRI and GBR."""
    input:
        vcf = "results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}.vcf.gz",
        sample = "results/realdata/{dataset}/LWK.sample",
        pilot_mask = paths["pilot_mask_hg38"],
    output:
        subset_vcf = "results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_LWK.vcf.gz",
        masked_vcf = "results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_LWK_pilot.vcf.gz",
    wildcard_constraints:
        chrom="|".join([str(i) for i in range(1, 23)]),  # explicitly define possible values
    shell:
        """
        bcftools view --samples-file {input.sample} -m 2 -M 2 -v snps {input.vcf} -Oz -o {output.subset_vcf}
        bcftools index {output.subset_vcf}
        bcftools view -R {input.pilot_mask} {output.subset_vcf} -Oz -o {output.masked_vcf}
        """


rule polarize_vcf:
    """Polarize VCF."""
    input:
        vcf = "results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_LWK.vcf.gz",
        polarize_vcf = paths["polarize_vcf"],
    params:
        ancestral=paths["human_ancestor"],
        chrom=lambda wildcards: wildcards.chrom,
    output:
        polarized_vcf = "results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_LWK_polarized.vcf",
        unploarized_sites = "results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_LWK_unpolarized.sites",
        flipped = "results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_LWK_flipped.sites"
    wildcard_constraints:
        chrom="|".join([str(i) for i in range(1, 23)]),  # explicitly define possible values
    shell:
        """
        python {input.polarize_vcf} -vcf {input.vcf} -fasta {params.ancestral}{params.chrom}.fa -output {output.polarized_vcf} -unpolarized {output.unploarized_sites} -flipped {output.flipped}
        """


rule polarize_vcf_pilot:
    """Polarize VCF."""
    input:
        vcf="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_YRI_GBR_pilot.vcf.gz",
        polarize_vcf=paths["polarize_vcf"],
    params:
        ancestral=paths["human_ancestor"],
        chrom=lambda wildcards: wildcards.chrom,
    output:
        polarized_vcf="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_YRI_GBR_pilot_polarized.vcf",
        unploarized_sites="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_YRI_GBR_pilot_unpolarized.sites",
        flipped="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_YRI_GBR_pilot_flipped.sites",
    wildcard_constraints:
        chrom="|".join([str(i) for i in range(1, 23)]),  # explicitly define possible values
    shell:
        """
        python {input.polarize_vcf} -vcf {input.vcf} -fasta {params.ancestral}{params.chrom}.fa -output {output.polarized_vcf} -unpolarized {output.unploarized_sites} -flipped {output.flipped}
        """


rule run_SINGER:
    """Estimate Ne from pi, run SINGER."""
    input:
        vcf="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_YRI_GBR_polarized.vcf",
        parallel_singer=str(paths["singer"] + "/parallel_singer"),
    params:
        vcf_pref="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_YRI_GBR_polarized",
        sub_size=2100000,
        nsamp=200,
        thin=20,
        mu=1.2e-8,
        ne=2e4,
        outpref="results/realdata/{dataset}/singer/yri_gbr_chr{chrom}",
    output:
        trees=expand(
            "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_{STEP}.trees",
            STEP=range(150, 200),
            allow_missing=True,
        ),
    wildcard_constraints:
        chrom="|".join([str(i) for i in range(1, 23)]),  # explicitly define possible values
    threads: 120
    shell:
        """
        {input.parallel_singer} -vcf {params.vcf_pref} -L {params.sub_size} -n {params.nsamp} -thin {params.thin} -polar 0.99 -num_cores {threads} -m {params.mu} -Ne {params.ne} -output {params.outpref}
        """


rule run_SINGER_pilot:
    """Estimate Ne from pi, run SINGER."""
    input:
        vcf="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_YRI_GBR_pilot_polarized.vcf",
        full_vcf="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_YRI_GBR_polarized.vcf",
        parallel_singer=str(paths["singer"] + "/parallel_singer"),
    params:
        vcf_pref="results/realdata/{dataset}/vcf_files/1000g_hg38_chr{chrom}_YRI_GBR_pilot_polarized",
        sub_size=2100000,
        nsamp=200,
        thin=20,
        mu=1.2e-8,
        ne=2e4,
        # outpref = "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot",
        outpref="results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_fullmu",
    output:
        trees=expand(
            "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_fullmu_{STEP}.trees",
            STEP=range(150, 200),
            allow_missing=True,
        ),
        # trees=expand(
        #     "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_{STEP}.trees",
        #     STEP = range(150, 200),
        #     allow_missing=True,
        # ),
    wildcard_constraints:
        chrom="|".join([str(i) for i in range(1, 23)]),  # explicitly define possible values
    threads: 120
    shell:
        """
#        a=$(bcftools stats {input.vcf} | grep "number of SNPs:" | awk '{{print $6}}')
#        b=$(bcftools stats {input.full_vcf} | grep "number of SNPs:" | awk '{{print $6}}')
#        m=1.2*10^-8
#        mu=$(echo "$m * ($a / $b)" | bc -l)
#        {input.parallel_singer} -vcf {params.vcf_pref} -L {params.sub_size} -n {params.nsamp} -thin {params.thin} -polar 0.99 -num_cores {threads} -m $mu -Ne {params.ne} -output {params.outpref}
        {input.parallel_singer} -vcf {params.vcf_pref} -L {params.sub_size} -n {params.nsamp} -thin {params.thin} -polar 0.99 -num_cores {threads} -m {params.mu} -Ne {params.ne} -output {params.outpref}
        """


rule subtree_tsk:
    """Extract subtree for population A."""
    input:
        trees=expand(
            "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_{STEP}.trees",
            STEP=range(150, 200),
            allow_missing=True,
        ),
    output:
        Atrees=expand(
            "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_{STEP}.tsz",
            STEP=range(150, 200),
            allow_missing=True,
        ),
    wildcard_constraints:
        chrom="|".join([str(i) for i in range(1, 23)]),  # explicitly define possible values
    run:
        for step in range(50):
            ts = tskit.load(input.trees[step])
            tszip.compress(ts, output.Atrees[step])


rule subtree_tsk_pilot:
    """Extract subtree for population A."""
    input:
        # trees=expand(
        #     "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_{STEP}.trees",
        #     STEP = range(150, 200),
        #     allow_missing=True,
        # ),
        trees=expand(
            "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_fullmu_{STEP}.trees",
            STEP=range(150, 200),
            allow_missing=True,
        ),
    output:
        # Atrees=expand(
        #     "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_{STEP}.tsz",
        #     STEP = range(150, 200),
        #     allow_missing=True,
        # ),
        Atrees=expand(
            "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_fullmu_{STEP}.tsz",
            STEP=range(150, 200),
            allow_missing=True,
        ),
    wildcard_constraints:
        chrom="|".join([str(i) for i in range(1, 23)]),  # explicitly define possible values
    run:
        for step in range(50):
            ts = tskit.load(input.trees[step])
            tszip.compress(ts, output.Atrees[step])


rule remove_intermediate_files:
    """Remove intermediate files."""
    input:
        Atrees=expand(
            "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_{STEP}.tsz",
            STEP=range(150, 200),
            allow_missing=True,
        ),
    params:
        path="results/realdata/{dataset}/singer/yri_gbr_chr{chrom}",
    wildcard_constraints:
        chrom="|".join([str(i) for i in range(1, 23)]),  # explicitly define possible values
    output:
        done="results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_done.log",
    shell:
        """
        rm {params.path}_*muts*txt
        rm {params.path}_*recombs*txt
        rm {params.path}_*branches*txt
        rm {params.path}_*nodes*txt
        for i in $(seq 0 200)
        do
            rm -r {params.path}_$i.trees
        done
        echo "Done" > {output.done}
        """


rule remove_intermediate_files_pilot:
    """Remove intermediate files."""
    input:
        # Atrees=expand(
        #     "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_{STEP}.tsz",
        #     STEP = range(150, 200),
        #     allow_missing=True,
        # ),
        Atrees=expand(
            "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_fullmu_{STEP}.tsz",
            STEP=range(150, 200),
            allow_missing=True,
        ),
    params:
        # path = "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot",
        path="results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_fullmu",
    wildcard_constraints:
        chrom="|".join([str(i) for i in range(1, 23)]),  # explicitly define possible values
    output:
        # done = "results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_done.log",
        done="results/realdata/{dataset}/singer/yri_gbr_chr{chrom}_pilot_fullmu_done.log",
    shell:
        """
        rm {params.path}_*muts*txt
        rm {params.path}_*recombs*txt
        rm {params.path}_*branches*txt
        rm {params.path}_*nodes*txt
        for i in $(seq 0 200)
        do
            rm -r {params.path}_$i.trees
        done
        echo "Done" > {output.done}
        """
