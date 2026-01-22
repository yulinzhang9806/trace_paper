#!python3

import numpy as np
import pandas as pd
import sys
import demes
import msprime
import tszip
import yaml
from tqdm import tqdm
from arg_hmm.utils import ARG_utils

# -------------- Simulate for inference comparison -------------
rule simulate_with_arc:
    input:
        model=ancient("results/simulations/models/{model}/{model}_{version}.yaml"),
    params:
        model=lambda wildcards: wildcards.model,
        version = lambda wildcards: wildcards.version,
        sequence_length=250e6,
        recombination_rate=1e-8,
    output:
        tsz="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.tsz",
        vcf="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.vcf",
        null_vcf="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_null.vcf",
        target_tsz="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_A_full.tsz",
        null_tsz="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_null_full.tsz",
        sample_json="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_samples.json",
    run:
        model = demes.load(str(input.model))
        demo = msprime.Demography.from_demes(model)
        pop_dict = {i.name: i.id for i in demo.populations}
        n = int(wildcards.n)
        seed = int(wildcards.seed)
        if "YRI" in pop_dict.keys():
            ts = msprime.sim_ancestry(
                samples = [
                    msprime.SampleSet(n, population=pop_dict["A"]),
                    msprime.SampleSet(n, population=pop_dict["YRI"]),
                    msprime.SampleSet(1, population=pop_dict["B"], time=1600),
                ],
                sequence_length=params.sequence_length,
                recombination_rate=params.recombination_rate,
                random_seed=seed,
                demography=demo,
                record_migrations=True,
            )
        else:
            ts = msprime.sim_ancestry(
                samples = [
                    msprime.SampleSet(n, population=pop_dict["A"]),
                    msprime.SampleSet(n, population=pop_dict["C"]),
                    msprime.SampleSet(1, population=pop_dict["B"], time=1600),
                ],
                sequence_length=params.sequence_length,
                recombination_rate=params.recombination_rate,
                random_seed=seed,
                demography=demo,
                record_migrations=True,
            )
        tszip.compress(ts, str(output.tsz))
        mts = msprime.sim_mutations(ts, rate=1.2e-8, random_seed=seed)
        with open(str(output.vcf), "w") as vcf:
            mts.write_vcf(vcf, contig_id=1)
        with open(str(output.sample_json), "w") as f:
            f.write('{\n\t"ingroup": [\n\t\t')
            for i in range(n - 1):
                f.write('"' + f"tsk_{i}" + '",\n\t\t')
            f.write('"' + f"tsk_{n-1}" + '"\n\t],\n\t"outgroup": [\n\t\t')
            for i in range(n, 2 * n - 1):
                f.write('"' + f"tsk_{i}" + '",\n\t\t')
            f.write('"' + f"tsk_{2*n-1}" + '"]\n')
            f.write("}")

        if "YRI" in pop_dict.keys():
            ts = msprime.sim_ancestry(
                samples = [
                    msprime.SampleSet(n, population=pop_dict["A"]),
                    msprime.SampleSet(n, population=pop_dict["YRI"]),
                    msprime.SampleSet(1, population=pop_dict["B"], time=1600),
                ],
                sequence_length=params.sequence_length,
                recombination_rate=params.recombination_rate,
                random_seed=seed,
                demography=demo,
            )
            moderns = np.concatenate([ts.samples(pop_dict["A"]), ts.samples(pop_dict["YRI"])])
            tszip.compress(
                ts.simplify(samples=moderns), str(output.target_tsz)
            )
            graph = demes.load(input.model)
            graph_dict = graph.asdict()
            graph_dict["pulses"][0]["proportions"] = [0]
            sim_graph = demes.Graph.fromdict(graph_dict)
            demo = msprime.Demography.from_demes(sim_graph)
            ts = msprime.sim_ancestry(
                samples = [
                    msprime.SampleSet(n, population=pop_dict["A"]),
                    msprime.SampleSet(n, population=pop_dict["YRI"]),
                    msprime.SampleSet(1, population=pop_dict["B"], time=1600),
                ],
                sequence_length=params.sequence_length,
                recombination_rate=params.recombination_rate,
                random_seed=seed,
                demography=demo,
            )
            mts = msprime.sim_mutations(ts, rate=1.2e-8, random_seed=seed)
            moderns = np.concatenate([ts.samples(pop_dict["A"]), ts.samples(pop_dict["YRI"])])
            tszip.compress(
                ts.simplify(samples=moderns), str(output.null_tsz)
            )
            with open(str(output.null_vcf), "w") as vcf:
                mts.write_vcf(vcf, contig_id=1)
        else:
            ts = msprime.sim_ancestry(
                samples = [
                    msprime.SampleSet(n, population=pop_dict["A"]),
                    msprime.SampleSet(n, population=pop_dict["C"]),
                    msprime.SampleSet(1, population=pop_dict["B"], time=1600),
                ],
                sequence_length=params.sequence_length,
                recombination_rate=params.recombination_rate,
                random_seed=seed,
                demography=demo,
            )
            moderns = np.concatenate([ts.samples(pop_dict["A"]), ts.samples(pop_dict["C"])])
            tszip.compress(
                ts.simplify(samples=moderns), str(output.target_tsz)
            )
            graph = demes.load(input.model)
            graph_dict = graph.asdict()
            graph_dict["pulses"][0]["proportions"] = [0]
            sim_graph = demes.Graph.fromdict(graph_dict)
            demo = msprime.Demography.from_demes(sim_graph)
            ts = msprime.sim_ancestry(
                samples = [
                    msprime.SampleSet(n, population=pop_dict["A"]),
                    msprime.SampleSet(n, population=pop_dict["C"]),
                    msprime.SampleSet(1, population=pop_dict["B"], time=1600),
                ],
                sequence_length=params.sequence_length,
                recombination_rate=params.recombination_rate,
                random_seed=seed,
                demography=demo,
            )
            mts = msprime.sim_mutations(ts, rate=1.2e-8, random_seed=seed)
            moderns = np.concatenate([ts.samples(pop_dict["A"]), ts.samples(pop_dict["C"])])
            tszip.compress(
                ts.simplify(samples=moderns), str(output.null_tsz)
            )
            with open(str(output.null_vcf), "w") as vcf:
                mts.write_vcf(vcf, contig_id=1)

rule extract_ghost_admix_arc:
    """Extract true ghost admixture events from tree-sequences.

    NOTE: this rule extracts the true ghost sections from the simple models...
    """
    input:
        tsz="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.tsz",
        model="results/simulations/models/{model}/{model}_{version}.yaml",
    params:
        model=lambda wildcards: wildcards.model,
        version=lambda wildcards: wildcards.version,
        n=lambda wildcards: wildcards.n,
        seed=lambda wildcards: wildcards.seed,
    output:
        ind_bed="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.indiv.bed",
        merged_bed="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.merged.bed",
    run:
        ts = tszip.decompress(str(input.tsz))
        # Read in the model and create a population-dictionary
        demo = msprime.Demography.from_demes(demes.load(input.model))
        pop_dict = {i.name: i.id for i in demo.populations}
        if params.model == "ooa_neanderthal5r19_comp" or params.version == "intro_mha":
            n = 2 * int(wildcards.n)
        else:
            n = int(wildcards.n)
        arg_utils = ARG_utils(
            total_sample_size=2 * n,
            afr_poplabel=pop_dict["A"],
            ghost_poplabel=pop_dict["B"],
        )
        arg_utils.add_tree_sequence(ts)
        if params.model == "ooa_neanderthal5r19_simp" and params.version == "intro_mha":
            arg_utils.extract_ghost_intro_all(
                from_pop=pop_dict["YRI"],
                to_pop=pop_dict["B"],
            )
        else:
            arg_utils.extract_ghost_intro_all(
                from_pop=pop_dict["A"],
                to_pop=pop_dict["B"],
            )
        arg_utils.write_bed_output(
            str(output.ind_bed), arg_utils.ne_seg, chrom=1, indinfo=True, cm=True
        )
        arg_utils.write_bed_output(
            str(output.merged_bed), arg_utils.ne_seg, chrom=1, indinfo=False, cm=True
        )

# --------------- IBDmix ----------------

rule run_ibdmix:
    input:
        vcf="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.vcf",
        vcf_null="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_null.vcf",
        generate_gt = str(paths["ibdmix_exe"] + "generate_gt"),
        ibdmix = str(paths["ibdmix_exe"] + "ibdmix"),
    output:
        altai_gt="results/hmm_results/inference_comparison/{model}/{version}/ibdmix/n{n}_seed{seed}_A.gt",
        altai_out="results/hmm_results/inference_comparison/{model}/{version}/ibdmix/n{n}_seed{seed}_A.ibdmix",
        null_gt="results/hmm_results/inference_comparison/{model}/{version}/ibdmix/n{n}_seed{seed}_null.gt",
        null_out="results/hmm_results/inference_comparison/{model}/{version}/ibdmix/n{n}_seed{seed}_null.ibdmix",
    params:
        n=lambda wildcards: wildcards.n,
        outpref="results/hmm_results/inference_comparison/{model}/{version}/ibdmix/n{n}_seed{seed}",
    shell:
        """
        n={params.n}
        altai=$(bcftools query -l {input.vcf} | tail -n 1)
        bcftools view -s $altai -Ov {input.vcf} > {params.outpref}_altai.vcf
        bcftools view -s $altai -Ov {input.vcf_null} > {params.outpref}_null_altai.vcf
        bcftools view -s ^$altai -Ov {input.vcf} > {params.outpref}_eur.vcf
        bcftools view -s ^$altai -Ov {input.vcf_null} > {params.outpref}_null_eur.vcf
        {input.generate_gt} -a {params.outpref}_altai.vcf -m {params.outpref}_eur.vcf -o {output.altai_gt}
        {input.generate_gt} -a {params.outpref}_null_altai.vcf -m {params.outpref}_null_eur.vcf -o {output.null_gt}
        {input.ibdmix} -g {output.altai_gt} --LOD-threshold 4.0 --minor-allele-count-threshold 1 --archaic-error 0.01 --modern-error-max 0.002 --modern-error-proportion 2 -o {output.altai_out}
        {input.ibdmix} -g {output.null_gt} --LOD-threshold 4.0 --minor-allele-count-threshold 1 --archaic-error 0.01 --modern-error-max 0.002 --modern-error-proportion 2 -o {output.null_out}
        """
        # eur_sample=$(for ((i=0; i<n; i++)); do echo "tsk_$i"; done | paste -sd,)
        # bcftools view -s $eur_sample -Ov {input.vcf} > {params.outpref}_eur.vcf
        # bcftools view -s $eur_sample -Ov {input.vcf_null} > {params.outpref}_null_eur.vcf
# ------------------------ HMMIX Processing -------------------------- #

rule create_outgroup_whole:
    """Get positions of variable SNPs in outgroup population."""
    input:
        vcf="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.vcf",
        vcf_null="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_null.vcf",
        ind_file="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_samples.json",
    output:
        outgroup="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_withARC_outgroup.txt",
        outgroup_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null_outgroup.txt",
    shell:
        """
        afr_samps=$(printf 'tsk_%d,' {{100..199}} | sed 's/,$//')
        bcftools view -s $afr_samps {input.vcf} | bcftools norm -m -any | bcftools view -v snps -c 1 | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%REF\n' > {output.outgroup}
        bcftools view -s $afr_samps {input.vcf_null} | bcftools norm -m -any | bcftools view -v snps -c 1 | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t%REF\n' > {output.outgroup_null}
        """


rule est_mutrate_whole:
    """Estimating mutation rate using SNPs in the outgroup population"""
    input:
        outgroup="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_withARC_outgroup.txt",
        outgroup_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null_outgroup.txt",
    output:
        mutrate="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_mutrate_withARC.bed",
        mutrate_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null_mutrate_withARC.bed",
    params:
        windowsize=1000000,
    shell:
        """
        hmmix mutation_rate -outgroup={input.outgroup} -window_size={params.windowsize} -out {output.mutrate}
        hmmix mutation_rate -outgroup={input.outgroup_null} -window_size={params.windowsize} -out {output.mutrate_null}
        """

rule create_ingroup:
    """Get unique SNPs in the target population."""
    input:
        vcf="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.vcf",
        vcf_null="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_null.vcf",
        ind_file="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_samples.json",
        outgroup="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_withARC_outgroup.txt",
        outgroup_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null_outgroup.txt",
    output:
        ingroup = expand(
            "results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_{ind}.txt",
            ind=range(10),
            allow_missing=True,
        ),
        ingroup_null = expand(
            "results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_{ind}.txt",
            ind=range(10),
            allow_missing=True,
        ),
    params:
        n=lambda wildcards: wildcards.n,
        seed=lambda wildcards: wildcards.seed,
        outpref="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}",
        outpref_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null",
    threads: 20
    shell:
        """
        eur_samps=$(printf 'tsk_%d,' {{0..99}} | sed 's/,$//')
        for samp in ${{eur_samps//,/ }};
        do
            bcftools view -s $samp -T ^{input.outgroup} {input.vcf}| bcftools norm -m -any | bcftools view -v snps -c 1 | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' | awk 'BEGIN {{OFS="\t"}} {{gsub("0", $3, $5); gsub("1", $4, $5); gsub("\\\\|", "", $5); print $1,$2,$3,$5}}' > {params.outpref}.$samp.txt &
        done
        wait
        for samp in ${{eur_samps//,/ }};
        do
            bcftools view -s $samp -T ^{input.outgroup_null} {input.vcf_null}| bcftools norm -m -any | bcftools view -v snps -c 1 | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\t[%GT]\n' | awk 'BEGIN {{OFS="\t"}} {{gsub("0", $3, $5); gsub("1", $4, $5); gsub("\\\\|", "", $5); print $1,$2,$3,$5}}' > {params.outpref_null}.$samp.txt &
        done
        wait
        """

rule train_hmmix:
    """Train hmmix."""
    input:
        obs="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_{ind}.txt",
        obs_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_{ind}.txt",
        mutrate="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_mutrate_withARC.bed",
        mutrate_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null_mutrate_withARC.bed",
    output:
        json="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_{ind}.trained.json",
        json_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_{ind}.trained.json",
    shell:
        """
        hmmix train -obs={input.obs} -mutrates={input.mutrate} -out={output.json} -haploid
        hmmix train -obs={input.obs_null} -mutrates={input.mutrate_null} -out={output.json_null} -haploid
        """

rule decode_hmmix:
    """Decode hmmix."""
    input:
        obs="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_{ind}.txt",
        obs_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_{ind}.txt",
        mutrate="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_mutrate_withARC.bed",
        mutrate_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null_mutrate_withARC.bed",
        json="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_{ind}.trained.json",
        json_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_{ind}.trained.json",
    output:
        out="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_{ind}.hap1.txt",
        out_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_{ind}.hap1.txt",
        out2="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_{ind}.hap2.txt",
        out2_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_{ind}.hap2.txt",
    params:
        outpref="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_{ind}",
        outpref_null="results/hmm_results/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_{ind}",
    shell:
        """
        hmmix decode -obs={input.obs} -mutrates={input.mutrate} -param={input.json} -out={params.outpref} -haploid
        hmmix decode -obs={input.obs_null} -mutrates={input.mutrate_null} -param={input.json_null} -out={params.outpref_null} -haploid
        """


# ------------------------ Sprime Processing  -------------------------- #


rule get_all_positions:
    """Get all of the positions for SNPs."""
    input:
        vcf="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.vcf",
        null_vcf="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_null.vcf",
    output:
        pos="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}.all.pos",
        pos_null="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_null.all.pos",
    shell:
        """
        bcftools view {input.vcf} -m 2 -M 2 -v snps | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' > {output.pos}
        bcftools view {input.null_vcf} -m 2 -M 2 -v snps | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' > {output.pos_null}
        """


rule get_Sprime_map:
    """Generate input for Sprime."""
    input:
        pos="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}.all.pos",
        pos_null="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_null.all.pos",
    params:
        n=lambda wildcards: wildcards.n,
        inds=lambda wildcards: analysis_input["inference_comparison"]["inds"],
    output:
        out_map="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}.map",
        outgroup="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_outgroup.sample",
        excl_sample="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_excl.sample",
        out_null_map="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_null.map",
    run:
        df = pd.read_csv(input.pos, sep="\t", names=["chrom", "pos", "ref", "alt"])
        df["id"] = df[["chrom", "pos"]].astype("str").agg(":".join, axis=1)
        df["genetic_dis"] = df["pos"] / 1e6
        df = df[["chrom", "id", "genetic_dis", "pos"]]
        df.to_csv(
            output.out_map, sep="\t", header=False, index=False, float_format="%.6f"
        )
        with open(str(output.outgroup), "w") as f:
            for i in range(int(params.n), 2 * int(params.n)):
                f.write(f"tsk_{i}\n")
        with open(str(output.excl_sample), "w") as f:
            for i in range(int(inds), int(params.n)):
                f.write(f"tsk_{i}\n")
            f.write(f"tsk_{2 * int(params.n)}")
        df = pd.read_csv(input.pos_null, sep="\t", names=["chrom", "pos", "ref", "alt"])
        df["id"] = df[["chrom", "pos"]].astype("str").agg(":".join, axis=1)
        df["genetic_dis"] = df["pos"] / 1e6
        df = df[["chrom", "id", "genetic_dis", "pos"]]
        df.to_csv(
            output.out_null_map, sep="\t", header=False, index=False, float_format="%.6f"
        )

rule run_Sprime:
    """Run Sprime."""
    input:
        vcf="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.vcf",
        vcf_null="results/hmm_results/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_null.vcf",
        outgroup="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_outgroup.sample",
        exclude_sample="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_excl.sample",
        in_map="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}.map",
        in_map_null="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_null.map",
        sprime="data/other_methods_exe/sprime.jar",
    params:
        GB=10,
        outpref="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_Sprime",
        outpref_null="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_null_Sprime",
    output:
        out="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_Sprime.score",
        out_null="results/hmm_results/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_null_Sprime.score",
    shell:
        """
        java -Xmx{params.GB}g -jar {input.sprime} gt={input.vcf} outgroup={input.outgroup} map={input.in_map} excludesamples={input.exclude_sample} out={params.outpref} minscore=1000
        java -Xmx{params.GB}g -jar {input.sprime} gt={input.vcf_null} outgroup={input.outgroup} map={input.in_map_null} excludesamples={input.exclude_sample} out={params.outpref_null} minscore=1000
        """

# # ------------------------ Gamma SMC Processing  -------------------------- #

# rule run_gamma_smc:
#     """Run gamma smc."""
#     input:
#         vcf = "results/simulations/outputs/{model}/{version}/n{n}_seed{seed}_{p}.vcf",
#         gamma_smc = str(paths["gamma_smc"] + "/bin/gamma_smc"),
#     output:
#         zst = "results/hmm_results/arg_infer/{model}/{version}/gammasmc/n{n}_seed{seed}_{p}.zst"
#     shell:
#         """
#         {input.gamma_smc} -i {input.vcf} -s 1000 -t 0.83 -o {output.zst}
#         """




