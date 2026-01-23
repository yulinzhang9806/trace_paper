#!python3

import demes
import msprime
import tszip
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm
from utils import ARG_utils

rule set_simple_model:
    input:
        simple_simulation="data/template_models/simple/simple_{model}.yaml",
    output:
        sim_yaml=str(paths["benchmark_simulations"]) + "/models/{model}/{model}_{version}.yaml",
    params:
        model=lambda wildcards: wildcards.model,
        archaic_split=lambda wildcards: simple_config["models"][wildcards.model][
            "versions"
        ][wildcards.version]["archaic_split"],
        migration_rate=lambda wildcards: simple_config["models"][wildcards.model][
            "versions"
        ][wildcards.version]["migration_rate"],
        migration_time=lambda wildcards: simple_config["models"][wildcards.model][
            "versions"
        ][wildcards.version]["migration_time"],
        growth_end_size=lambda wildcards: simple_config["models"][wildcards.model][
            "versions"
        ][wildcards.version]["growth_n"],
        bottleneck_start=lambda wildcards: simple_config["models"][wildcards.model][
            "versions"
        ][wildcards.version]["bottleneck_start"],
        bottleneck_time=lambda wildcards: simple_config["models"][wildcards.model][
            "versions"
        ][wildcards.version]["bottleneck_time"],
        bottleneck_size=lambda wildcards: simple_config["models"][wildcards.model][
            "versions"
        ][wildcards.version]["bottleneck_size"],
    run:
        graph = demes.load(input.simple_simulation)
        graph_dict = graph.asdict()
        if params.model in ["ooa_neanderthal5r19_simp", "ooa_neanderthal5r19_comp"]:
            graph_dict["demes"][1]["start_time"] = params.archaic_split
            graph_dict["pulses"][0]["time"] = params.migration_time
            graph_dict["pulses"][0]["proportions"] = [params.migration_rate]
            if params.model in ["ooa_neanderthal5r19_simp", "ooa_neanderthal5r19_comp"]:
                if params.migration_time > params.bottleneck_start:
                    graph_dict["pulses"][0]["dest"] = "YRI"
                graph_dict["demes"][2]["start_time"] = params.bottleneck_start
                graph_dict["demes"][2]["epochs"][0]["end_time"] = params.bottleneck_time
                graph_dict["demes"][2]["epochs"][0]["start_size"] = params.bottleneck_size
                graph_dict["demes"][2]["epochs"][0]["end_size"] = params.bottleneck_size
        elif params.model == "split_multipulse":
            graph_dict["demes"][1]["start_time"] = params.archaic_split
            graph_dict["pulses"][0]["time"] = params.migration_time[1]
            graph_dict["pulses"][0]["proportions"] = [params.migration_rate[1]]
            graph_dict["pulses"][1]["time"] = params.migration_time[0]
            graph_dict["pulses"][1]["proportions"] = [params.migration_rate[0]]
            graph_dict["demes"][2]["start_time"] = params.migration_time[1] + 369
        else:
            graph_dict["demes"][1]["start_time"] = params.archaic_split
            graph_dict["pulses"][0]["time"] = params.migration_time[0]
            graph_dict["pulses"][0]["proportions"] = [params.migration_rate[0]]
            graph_dict["demes"][2]["start_time"] = params.bottleneck_start
            if float(params.migration_time[0]) >= 1725:
                graph_dict["demes"][2]["start_time"] = params.migration_time[0] + 369
            graph_dict["demes"][2]["epochs"][0]["end_time"] = params.bottleneck_time
            graph_dict["demes"][2]["epochs"][1]["end_size"] = params.growth_end_size
            graph_dict["demes"][2]["epochs"][0]["start_size"] = params.bottleneck_size
            graph_dict["demes"][2]["epochs"][0]["end_size"] = params.bottleneck_size
        sim_graph = demes.Graph.fromdict(graph_dict)
        demes.dump(sim_graph, str(output.sim_yaml))


rule sim_simple_model:
    """Simulate a simple model."""
    input:
        model=str(paths["benchmark_simulations"]) + "/models/{model}/{model}_{version}.yaml",
    output:
        tsz=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}.tsz",
        target_tsz=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_A.tsz",
        target_both_tsz=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_full.tsz",
        null_tsz=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_null.tsz",
        null_sub_tsz=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_null_A.tsz",
        vcf=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}.vcf",
        Asample=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_A.sample",
        outsample=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_outgroup.sample",
        sample_json=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_samples.json",
    params:
        model=lambda wildcards: wildcards.model,
        version = lambda wildcards: wildcards.version,
        sequence_length=lambda wildcards: simple_config["sequence_length"],
        recombination_rate=lambda wildcards: simple_config["recombination_rate"],
        null_vcf=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_null.vcf",
    run:
        model = demes.load(str(input.model))
        demo = msprime.Demography.from_demes(model)
        pop_dict = {i.name: i.id for i in demo.populations}
        print(pop_dict)
        n = int(wildcards.n)
        seed = int(wildcards.seed)
        # simulate with migration info
        ts = msprime.sim_ancestry(
            {"A": n, "C": n},
            sequence_length=params.sequence_length,
            recombination_rate=params.recombination_rate,
            random_seed=seed,
            demography=demo,
            record_migrations=True,
        )
        tszip.compress(ts, str(output.tsz))

        # record vcf and sample information
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
        with open(str(output.Asample), "w") as f:
            for i in range(n):
                f.write(f"tsk_{i}\n")
        with open(str(output.outsample), "w") as f:
            for i in range(n, 2 * n):
                f.write(f"tsk_{i}\n")

        # simulate without migration for decapitate to work
        ts = msprime.sim_ancestry(
            {"A": n, "C": n},
            sequence_length=params.sequence_length,
            recombination_rate=params.recombination_rate,
            random_seed=seed,
            demography=demo,
        )
        tszip.compress(
            ts.simplify(samples=ts.samples(pop_dict["A"])), str(output.target_tsz)
        )
        tszip.compress(
            ts, str(output.target_both_tsz)
        )
        graph = demes.load(input.model)
        graph_dict = graph.asdict()
        graph_dict["pulses"][0]["proportions"] = [0]
        sim_graph = demes.Graph.fromdict(graph_dict)
        demo = msprime.Demography.from_demes(sim_graph)
        ts = msprime.sim_ancestry(
            {"A": n, "C": n},
            sequence_length=params.sequence_length,
            recombination_rate=params.recombination_rate,
            random_seed=seed,
            demography=demo,
        )
        tszip.compress(
            ts.simplify(samples=ts.samples(pop_dict["A"])), str(output.null_sub_tsz)
        )
        tszip.compress(
            ts, str(output.null_tsz)
        )
        mts = msprime.sim_mutations(ts, rate=1.2e-8, random_seed=seed)
        with open(str(params.null_vcf), "w") as vcf:
            mts.write_vcf(vcf, contig_id=1)
            


rule extract_ghost_admix_simple:
    """Extract true archaic admixture events from tree-sequences."""
    input:
        tsz=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}.tsz",
        model=str(paths["benchmark_simulations"]) + "/models/{model}/{model}_{version}.yaml",
    params:
        model=lambda wildcards: wildcards.model,
        version=lambda wildcards: wildcards.version,
        n=lambda wildcards: wildcards.n,
        seed=lambda wildcards: wildcards.seed,
    output:
        ind_bed=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}.indiv.bed",
        merged_bed=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}.merged.bed",
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
                from_pop=pop_dict["C"],
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
