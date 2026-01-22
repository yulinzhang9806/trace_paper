#!python3

import numpy as np
import pandas as pd
import math
#from workflow.scripts.ARGweaver_utils import ARGweaver_related_utils, ARGweaver_to_ts
import tskit
import tszip
import json

# ------------------------ Run Relate -------------------------- #
rule get_A_subvcf:
    """Get a vcf containing only population A (target group)."""
    input:
        vcf="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}.vcf",
        A_sample="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_A.sample",
        out_sample="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_outgroup.sample",
    params:
        total_sample="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_A_total.sample"
    output:
        Fvcf="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_A_full.vcf",
    shell:
        """
        cat {input.A_sample} {input.out_sample} > {params.total_sample}
        bcftools view {input.vcf} -S {params.total_sample} -v snps -m 2 -M 2 -c 1 -Ov -o {output.Fvcf}
        rm -r {params.total_sample}
        """


rule get_outgroup_subvcf:
    """Get a vcf containing only outgroup."""
    input:
        vcf="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_null.vcf",
        null_sample="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_A.sample",
        out_sample="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_outgroup.sample",
    params:
        total_sample="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_null_total.sample"
    output:
        Fvcf="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_out_full.vcf",
    shell:
        """
        cat {input.null_sample} {input.out_sample} > {params.total_sample}
        bcftools view {input.vcf} -S {params.total_sample} -v snps -m 2 -M 2 -c 1 -Ov -o {output.Fvcf}
        rm -r {params.total_sample}
        """

rule relate_input:
    """Run input processing to convert file format from vcf to haps sample."""
    input:
        AFvcf="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_A_full.vcf",
        outFvcf="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_out_full.vcf",
        relate_input=str(paths["relate"] + "/bin/RelateFileFormats"),
    params:
        AFpref="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_A_full",
        outFpref="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_out_full",
    output:
        AFhaps="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full.haps",
        AFsample="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full.sample",
        outFhaps="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full.haps",
        outFsample="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full.sample",
    shell:
        """
        {input.relate_input} --mode ConvertFromVcf --haps {output.AFhaps} --sample {output.AFsample} --chr 1 -i {params.AFpref}
        {input.relate_input} --mode ConvertFromVcf --haps {output.outFhaps} --sample {output.outFsample} --chr 1 -i {params.outFpref}
        """


rule get_all_positions:
    """Get all of the positions for SNPs."""
    input:
        vcf="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}.vcf",
        null_vcf="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_null.vcf",
    output:
        pos="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}.all.pos",
        null_pos="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out.all.pos",
    shell:
        """
        bcftools view {input.vcf} -m 2 -M 2 -v snps | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' > {output.pos}
        bcftools view {input.null_vcf} -m 2 -M 2 -v snps | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' > {output.null_pos}
        """


rule genetic_map_poplabel:
    """Generate genetic map for relate."""
    input:
        pos="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}.all.pos",
        null_pos="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out.all.pos",
        Asample="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_A.sample",
        outsample="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_outgroup.sample",
    output:
        A_map="results/hmm_results/arg_infer/{model}/{version}/relate/relate_n{n}_seed{seed}_A.map",
        null_map="results/hmm_results/arg_infer/{model}/{version}/relate/relate_n{n}_seed{seed}_out.map",
        Fullpoplabel="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_full.poplabel",
    run:
        df = pd.read_csv(input.pos, sep="\t", names=["chrom", "pos", "ref", "alt"])
        df["Rate(cM/Mb)"] = 1
        df["Map(cM)"] = df["pos"] / 1e6
        df = df[["pos", "Rate(cM/Mb)", "Map(cM)"]]
        df.columns = ["Position(bp)", "Rate(cM/Mb)", "Map(cM)"]
        df.to_csv(
            output.A_map, sep="\t", header=False, index=False, float_format="%.6f"
        )
        df = pd.read_csv(input.null_pos, sep="\t", names=["chrom", "pos", "ref", "alt"])
        df["Rate(cM/Mb)"] = 1
        df["Map(cM)"] = df["pos"] / 1e6
        df = df[["pos", "Rate(cM/Mb)", "Map(cM)"]]
        df.columns = ["Position(bp)", "Rate(cM/Mb)", "Map(cM)"]
        df.to_csv(
            output.null_map, sep="\t", header=False, index=False, float_format="%.6f"
        )
        df = pd.read_csv(
            input.Asample, sep="\t", names=["sample", "population", "group"]
        )
        df["sex"] = "NA"
        df["population"] = "A"
        df["group"] = "A"
        df1 = pd.read_csv(
            input.outsample, sep="\t", names=["sample", "population", "group"]
        )
        df1["sex"] = "NA"
        df1["population"] = "outgroup"
        df1["group"] = "outgroup"
        dff_c = pd.concat([df, df1])
        dff_c.to_csv(output.Fullpoplabel, sep="\t", index=False)
        print("Messages", file=sys.stderr)


rule run_relate:
    """Run Relate."""
    input:
        AFhaps="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full.haps",
        AFsample="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full.sample",
        outFhaps="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full.haps",
        outFsample="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full.sample",
        relate=str(paths["relate"] + "/scripts/RelateParallel/RelateParallel.sh"),
        Ageneticmap="results/hmm_results/arg_infer/{model}/{version}/relate/relate_n{n}_seed{seed}_A.map",
        nullgeneticmap="results/hmm_results/arg_infer/{model}/{version}/relate/relate_n{n}_seed{seed}_out.map",
    params:
        mu=1.2e-8,
        ne=1e4,
        AFprefix="n{n}_seed{seed}_A_full",
        outFprefix="n{n}_seed{seed}_out_full",
        Ageneticmap="relate_n{n}_seed{seed}_A.map",
        nullgeneticmap="relate_n{n}_seed{seed}_out.map",
        seed=lambda wildcards: wildcards.seed,
        workingdir="results/hmm_results/arg_infer/{model}/{version}/relate",
    output:
        AFanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full.anc",
        AFmut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full.mut",
        outFanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full.anc",
        outFmut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full.mut",
    threads: 10
    shell:
        """
        cd {params.workingdir}
        {input.relate} --haps {params.AFprefix}.haps --sample {params.AFprefix}.sample --map {params.Ageneticmap} --seed {params.seed} -m {params.mu} -N {params.ne} -o {params.AFprefix} --threads {threads}
        {input.relate} --haps {params.outFprefix}.haps --sample {params.outFprefix}.sample --map {params.nullgeneticmap} --seed {params.seed} -m {params.mu} -N {params.ne} -o {params.outFprefix} --threads {threads}
        """


rule run_reestimation:
    """Run Relate."""
    input:
        AFanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full.anc",
        AFmut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full.mut",
        Fullpoplabel="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_full.poplabel",
        outFanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full.anc",
        outFmut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full.mut",
        relate=str(
            paths["relate"]
            + "/scripts/EstimatePopulationSize/EstimatePopulationSize.sh"
        ),
        relate_split=str(paths["relate"] + "/bin/RelateExtract"),
    params:
        mu=1.2e-8,
        Aprefix="n{n}_seed{seed}_A",
        AFprefix="n{n}_seed{seed}_A_full",
        popAprefix="n{n}_seed{seed}_A_popsize",
        popAFprefix="n{n}_seed{seed}_A_full_popsize",
        outprefix="n{n}_seed{seed}_out",
        outFprefix="n{n}_seed{seed}_out_full",
        popoutprefix="n{n}_seed{seed}_out_popsize",
        popoutFprefix="n{n}_seed{seed}_out_full_popsize",
	Apoplabel="n{n}_seed{seed}_A.poplabels",
        Fullpoplabel="n{n}_seed{seed}_full.poplabel",
        seed=lambda wildcards: wildcards.seed,
        workingdir="results/hmm_results/arg_infer/{model}/{version}/relate",
    output:
        Aanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_popsize.anc.gz",
        Amut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_popsize.mut.gz",
        AFanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full_popsize.anc.gz",
        AFmut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full_popsize.mut.gz",
        outanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_popsize.anc.gz",
        outmut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_popsize.mut.gz",
        outFanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full_popsize.anc.gz",
        outFmut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full_popsize.mut.gz",
    threads: 10
    shell:
        """
        cd {params.workingdir}
        {input.relate_split} --mode SubTreesForSubpopulation --anc {params.AFprefix}.anc --mut {params.AFprefix}.mut --poplabels {params.Fullpoplabel} --pop_of_interest A --output {params.Aprefix}
        {input.relate_split} --mode SubTreesForSubpopulation --anc {params.outFprefix}.anc --mut {params.outFprefix}.mut --poplabels {params.Fullpoplabel} --pop_of_interest A --output {params.outprefix}
        {input.relate} -i {params.Aprefix} -m {params.mu} --poplabels {params.Apoplabel} --seed {params.seed} -o {params.popAprefix} --threads {threads}
        {input.relate} -i {params.AFprefix} -m {params.mu} --poplabels {params.Fullpoplabel} --seed {params.seed} -o {params.popAFprefix} --threads {threads}
        {input.relate} -i {params.outprefix} -m {params.mu} --poplabels {params.Apoplabel} --seed {params.seed} -o {params.popoutprefix} --threads {threads}
        {input.relate} -i {params.outFprefix} -m {params.mu} --poplabels {params.Fullpoplabel} --seed {params.seed} -o {params.popoutFprefix} --threads {threads}
        """


rule relate_to_ts:
    """Transfer Relate output to ts trees."""
    input:
        Aanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_popsize.anc.gz",
        Amut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_popsize.mut.gz",
        AFanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full_popsize.anc.gz",
        AFmut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full_popsize.mut.gz",
        outanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_popsize.anc.gz",
        outmut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_popsize.mut.gz",
        outFanc="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full_popsize.anc.gz",
        outFmut="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full_popsize.mut.gz",
        relate_ts=str(paths["relate_lib"] + "/bin/Convert"),
    params:
        workingdir="results/hmm_results/arg_infer/{model}/{version}/relate",
        Apref="n{n}_seed{seed}_A",
        AFpref="n{n}_seed{seed}_A_full",
        outpref="n{n}_seed{seed}_out",
        outFpref="n{n}_seed{seed}_out_full",
    output:
        Ats_tree="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A.trees",
        AFts_tree="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full.trees",
        outts_tree="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out.trees",
        outFts_tree="results/hmm_results/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full.trees",
    shell:
        """
        cd {params.workingdir}
        {input.relate_ts} --mode ConvertToTreeSequence --anc {params.Apref}_popsize.anc --mut {params.Apref}_popsize.mut -o {params.Apref}
        {input.relate_ts} --mode ConvertToTreeSequence --anc {params.AFpref}_popsize.anc --mut {params.AFpref}_popsize.mut -o {params.AFpref}
        {input.relate_ts} --mode ConvertToTreeSequence --anc {params.outpref}_popsize.anc --mut {params.outpref}_popsize.mut -o {params.outpref}
        {input.relate_ts} --mode ConvertToTreeSequence --anc {params.outFpref}_popsize.anc --mut {params.outFpref}_popsize.mut -o {params.outFpref}
        """

# ------------------------ Run SINGER -------------------------- #

rule run_SINGER_A:
    """Estimate Ne from pi, run SINGER."""
    input:
        vcf="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_A_full.vcf",
        parallel_singer = str(paths["singer"] + "/parallel_singer"),
    params:
        genome_length = 50000000,
        pi_pref = "n{n}_seed{seed}_A_full",
        vcf_pref = "results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_A_full",
        sub_size = 2000000,
        nsamp = 100,
        thin = 20, 
        mu = 1.2e-8,
        outpref = "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_full",
    output:
        trees=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_full_{STEP}.trees",
            STEP = range(90, 100),
            allow_missing=True,
        ),
    threads: 25
    shell:
        """
        vcftools --vcf {input.vcf} --window-pi {params.genome_length} --out {params.pi_pref}
        ane=$(echo "pi=`tail -1 {params.pi_pref}.windowed.pi | cut -f 5`; scale=10; (pi/4.8)*10*10*10*10*10*10*10*10" | bc)
        rm -r {params.pi_pref}*
        {input.parallel_singer} -vcf {params.vcf_pref} -L {params.sub_size} -n {params.nsamp} -thin {params.thin} -polar 0.99 -num_cores {threads} -m {params.mu} -ratio 0.83 -Ne $ane -output {params.outpref}
        """

rule run_SINGER_null:
    """Estimate Ne from pi, run SINGER."""
    input:
        vcf="results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_out_full.vcf",
        parallel_singer = str(paths["singer"] + "/parallel_singer"),
    params:
        genome_length = 50000000,
        pi_pref = "n{n}_seed{seed}_out_full",
        vcf_pref = "results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_out_full",
        sub_size = 2000000,
        nsamp = 100,
        thin = 20, 
        mu = 1.2e-8,
        outpref = "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_full",
    output:
        trees=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_full_{STEP}.trees",
            STEP = range(90, 100),
            allow_missing=True,
        ),
    threads: 25
    shell:
        """
        vcftools --vcf {input.vcf} --window-pi {params.genome_length} --out {params.pi_pref}
        one=$(echo "pi=`tail -1 {params.pi_pref}.windowed.pi | cut -f 5`; scale=10; (pi/4.8)*10*10*10*10*10*10*10*10" | bc)
        rm -r {params.pi_pref}*
        {input.parallel_singer} -vcf {params.vcf_pref} -L {params.sub_size} -n {params.nsamp} -thin {params.thin} -polar 0.99 -num_cores {threads} -m {params.mu} -ratio 0.83 -Ne $one -output {params.outpref}
        """

rule subtree_tsk:
    """Extract subtree for population A."""
    input:
        trees=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_full_{STEP}.trees",
            STEP = range(90, 100),
            allow_missing=True,
        ),
        out_trees=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_full_{STEP}.trees",
            STEP = range(90, 100),
            allow_missing=True,
        ),
    params:
        n = lambda wildcards: wildcards.n,
    output:
        Atrees=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_{STEP}.tsz",
            STEP = range(90, 100),
            allow_missing=True,
        ),
        out_trees=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_{STEP}.tsz",
            STEP = range(90, 100),
            allow_missing=True,
        ),
        Acompressed=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_full_{STEP}.tsz",
            STEP = range(90, 100),
            allow_missing=True,
        ),
        out_compressed=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_full_{STEP}.tsz",
            STEP = range(90, 100),
            allow_missing=True,
        ),
    run:
        n = 2 * int(params.n)
        for step in range(10):
            ts = tskit.load(input.trees[step])
            tszip.compress(ts, output.Acompressed[step])
            subtree_A = ts.simplify(samples=np.array([i for i in range(n)]))
            tszip.compress(subtree_A, output.Atrees[step])
            ts = tskit.load(input.out_trees[step])
            tszip.compress(ts, output.out_compressed[step])
            subtree_out = ts.simplify(samples=np.array([i for i in range(n)]))
            tszip.compress(subtree_out, output.out_trees[step])

rule remove_intermediate_files:
    """Remove intermediate files."""
    input:
        Atrees=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_{STEP}.tsz",
            STEP = range(90, 100),
            allow_missing=True,
        ),
        out_trees=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_{STEP}.tsz",
            STEP = range(90, 100),
            allow_missing=True,
        ),
        Acompressed=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_full_{STEP}.tsz",
            STEP = range(90, 100),
            allow_missing=True,
        ),
        out_compressed=expand(
            "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_full_{STEP}.tsz",
            STEP = range(90, 100),
            allow_missing=True,
        ),
    params:
        path = "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}",
    output:
        done = "results/hmm_results/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_done.log",
    shell:
        """
        rm -r {params.path}_A*txt
	    rm -r {params.path}_out*txt
        for i in $(seq 0 80)
        do
            rm -r {params.path}_A_full_$i.trees
            rm -r {params.path}_out_full_$i.trees
        done
        for i in $(seq 91 99)
        do
            rm -r {params.path}_A_full_$i.trees
            rm -r {params.path}_out_full_$i.trees
        done
        echo "Done" > {output.done}
        """
