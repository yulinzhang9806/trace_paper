#!python3

import numpy as np
import pandas as pd
import tskit
import tszip
import msprime
import demes
from utils import *

rule simulate_msprime:
    input:
        deme_yaml = "data/Demography1_{m}.yaml"
    output:
        tsz = "results/{m}_model/msprime/n100_seed{seed}.tsz",
        vcf = "results/{m}_model/msprime/n100_seed{seed}.vcf",
        nea_bed = "results/{m}_model/msprime/n100_seed{seed}.NEA.indiv.bed",
        den_bed = "results/{m}_model/msprime/n100_seed{seed}.DEN.indiv.bed",
        ghost_bed = "results/{m}_model/msprime/n100_seed{seed}.Ghost.indiv.bed",
        json = "results/{m}_model/msprime/n100_seed{seed}_samples.json",
    wildcard_constraints:
        seed="|".join([str(x) for x in config["seeds"]]),
    params:
        n = 100,
        sequence_length = 50e6,
        recombination_rate = 1e-8,
        seed = lambda wildcards: int(wildcards.seed),
    run:
        model = demes.load(input.deme_yaml)
        demo = msprime.Demography.from_demes(model)
        pop_dict = {i.name: i.id for i in demo.populations}
        n = params.n
        s = params.seed
        ts = msprime.sim_ancestry(
            samples = [
                msprime.SampleSet(n, population=pop_dict["NonAfrican"]),
                msprime.SampleSet(n, population=pop_dict["African"]),
                msprime.SampleSet(1, population=pop_dict["Seq_NEA"], time=1600),
                msprime.SampleSet(1, population=pop_dict["Seq_DEN"], time=1500),
            ],
            sequence_length=params.sequence_length,
            recombination_rate=params.recombination_rate,
            random_seed=s,
            demography=demo,
            record_migrations = True,
        )
        mts = msprime.sim_mutations(ts, rate=1.2e-8, random_seed=s)
        with open(output.vcf, "w") as vcf:
            mts.write_vcf(vcf, contig_id=1)
        tszip.compress(ts, output.tsz)
        arg_utils = ARG_utils(
            total_sample_size=4 * n + 4,
            afr_poplabel=pop_dict["NonAfrican"],
            ghost_poplabel=pop_dict["Intro_NEA"],
        )
        arg_utils.add_tree_sequence(ts)
        arg_utils.extract_ghost_intro_all(
            from_pop=pop_dict["NonAfrican"],
            to_pop=pop_dict["Intro_NEA"],
        )
        arg_utils.write_bed_output(
            output.nea_bed, arg_utils.ne_seg, chrom=1, indinfo=True, cm=True
        )
        arg_utils.ne_seg = {i:[] for i in range(arg_utils.total_sample_size)}
        arg_utils.extract_ghost_intro_all(
            from_pop=pop_dict["NonAfrican"],
            to_pop=pop_dict["Intro_DEN"],
        )
        arg_utils.write_bed_output(
            output.den_bed, arg_utils.ne_seg, chrom=1, indinfo=True, cm=True
        )
        arg_utils.ne_seg = {i:[] for i in range(arg_utils.total_sample_size)}
        arg_utils.extract_ghost_intro_all(
            from_pop=pop_dict["African"],
            to_pop=pop_dict["Ghost"],
        )
        arg_utils.write_bed_output(
            output.ghost_bed, arg_utils.ne_seg, chrom=1, indinfo=True, cm=True
        )
        with open(output.json, "w") as f:
            f.write('{\n\t"ingroup": [\n\t\t')
            for i in range(n - 1):
                f.write('"' + f"tsk_{i}" + '",\n\t\t')
            f.write('"' + f"tsk_{n-1}" + '"\n\t],\n\t"outgroup": [\n\t\t')
            for i in range(n, 2 * n - 1):
                f.write('"' + f"tsk_{i}" + '",\n\t\t')
            f.write('"' + f"tsk_{2*n-1}" + '"]\n')
            f.write("}")
    
rule make_input_vcf:
    input:
        vcf = "results/{m}_model/msprime/n100_seed{seed}.vcf",
    output:
        bcf = "results/{m}_model/msprime/n100_seed{seed}.bcf",
        modern_vcf = "results/{m}_model/msprime/n100_seed{seed}_modern.vcf",
        arc_bcf = "results/{m}_model/msprime/n100_seed{seed}_arc.bcf",
        altai_vcf = "results/{m}_model/msprime/n100_seed{seed}_altai.vcf",
        den_vcf = "results/{m}_model/msprime/n100_seed{seed}_den.vcf",
    shell:
        """
        altai=$(bcftools query -l {input.vcf} | tail -n 2 | head -n 1)
        den=$(bcftools query -l {input.vcf} | tail -n 1)
        afr_samps=$(printf 'tsk_%d,' {{100..199}} | sed 's/,$//')
        eur_samps=$(printf 'tsk_%d,' {{0..99}} | sed 's/,$//')
        bcftools view -s "$altai" -Ov {input.vcf} -o {output.altai_vcf}
        bcftools view -s "$den" -Ov {input.vcf} -o {output.den_vcf}
        bcftools view -s "^$altai,$den" -Ov {input.vcf} -o {output.modern_vcf}
        bcftools view -s "$altai,$den" -Ob {input.vcf} -o {output.arc_bcf}
        bcftools view {input.vcf} -Ob -o {output.bcf}
        bcftools index {output.arc_bcf}
        bcftools index {output.bcf}
        """

########HMMIX############
rule create_outgroup_ingroup:
    input:
        vcf = "results/{m}_model/msprime/n100_seed{seed}_modern.vcf",
    output:
        outgroup = "results/{m}_model/hmmix/n100_seed{seed}.outgroup.txt",
        mutrate = "results/{m}_model/hmmix/n100_seed{seed}_mutrate.bed",
        ingroup = expand(
            "results/{m}_model/hmmix/n100_seed{seed}.tsk_{samp}.txt",
            samp=range(100),
            allow_missing=True,
        )
    shell:
        """
        afr_samps=$(printf 'tsk_%d,' {{100..199}} | sed 's/,$//')
        bcftools view -s "$afr_samps" {input.vcf} \
            | bcftools norm -m -any \
            | bcftools view -v snps -c 1 \
            | bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\n' > {output.outgroup}
        hmmix mutation_rate -outgroup={output.outgroup} -window_size=1000000 -out={output.mutrate}
        eur_samps=$(printf 'tsk_%d,' {{0..99}} | sed 's/,$//')
        for samp in $(echo "$eur_samps" | tr ',' ' ')
        do
            outfile="results/{wildcards.m}_model/hmmix/n100_seed{wildcards.seed}.$samp.txt"
            bcftools view -s "$samp" -T ^{output.outgroup} {input.vcf} \
                | bcftools norm -m -any \
                | bcftools view -v snps -c 1 \
                | bcftools query -f '%CHROM\\t%POS\\t%REF\\t%ALT\\t[%GT]\\n' \
                | awk 'BEGIN {{OFS="\\t"}} {{gsub("0", $3, $5); gsub("1", $4, $5); gsub("\\\\|", "", $5); print $1, $2, $3, $5}}' \
                > "$outfile"
        done
        """

rule combine_across_seeds:
    input:
        outgroup = expand(
            "results/{m}_model/hmmix/n100_seed{s}.outgroup.txt",
            s=config["seeds"],
            allow_missing=True,
        ),
        ingroup = expand(
            "results/{m}_model/hmmix/n100_seed{s}.tsk_{samp}.txt",
            s=config["seeds"],
            samp=range(100),
            allow_missing=True,
        )
    output:
        combined_outgroup = "results/{m}_model/hmmix/total_outgroup.txt",
        combined_ingroup = expand("results/{m}_model/hmmix/tsk_{samp}.totobs.txt", samp=range(100), allow_missing=True),
        combined_mutrate = "results/{m}_model/hmmix/total_mutrate.bed",
    params:
        seeds = " ".join([str(x) for x in config["seeds"]]),
    shell:
        """
        for seed in {params.seeds}
        do
            sed "s/^1\\t/$seed\\t/g" results/{wildcards.m}_model/hmmix/n100_seed$seed.outgroup.txt >> {output.combined_outgroup}
            for samp in $(seq 0 99)
            do
                sed "s/^1\\t/$seed\\t/g" results/{wildcards.m}_model/hmmix/n100_seed$seed.tsk_$samp.txt >> results/{wildcards.m}_model/hmmix/tsk_$samp.totobs.txt
            done
        done
        hmmix mutation_rate -outgroup={output.combined_outgroup} -window_size=1000000 -out={output.combined_mutrate}
        """

rule hmmix_train:
    input:
        combined_ingroup = "results/{m}_model/hmmix/tsk_{samp}.totobs.txt",
        combined_mutrate = "results/{m}_model/hmmix/total_mutrate.bed",
    output:
        trained_params = "results/{m}_model/hmmix/tsk_{samp}.trained.json",
    shell:
        """
        hmmix train -obs={input.combined_ingroup} -mutrates={input.combined_mutrate} -out={output.trained_params} -haploid
        """

rule hmmix_decode:
    input:
        trained_params = "results/{m}_model/hmmix/tsk_{samp}.trained.json",
        obs = "results/{m}_model/hmmix/n100_seed{seed}.tsk_{samp}.txt",
        mutrate = "results/{m}_model/hmmix/n100_seed{seed}_mutrate.bed",
        archaic_bcf = "results/{m}_model/msprime/n100_seed{seed}_arc.bcf",
    output:
        decoded1 = "results/{m}_model/hmmix/n100_seed{seed}.tsk_{samp}.hap1.txt",
        decoded2 = "results/{m}_model/hmmix/n100_seed{seed}.tsk_{samp}.hap2.txt",
    params:
        outpref = "results/{m}_model/hmmix/n100_seed{seed}.tsk_{samp}"
    shell:
        """
        hmmix decode -obs={input.obs} -mutrates={input.mutrate} -param={input.trained_params} -out={params.outpref} -haploid -admixpop={input.archaic_bcf}
        """

########IBDmix############

rule IBDmix:
    input:
        generate_gt = paths["generate_gt"],
        ibdmix = paths["ibdmix"],
        altai_vcf = "results/{m}_model/msprime/n100_seed{seed}_altai.vcf",
        den_vcf = "results/{m}_model/msprime/n100_seed{seed}_den.vcf",
        modern_vcf = "results/{m}_model/msprime/n100_seed{seed}_modern.vcf",
    output:
        altai_gt = "results/{m}_model/ibdmix/n100_seed{seed}_altai.gt",
        den_gt = "results/{m}_model/ibdmix/n100_seed{seed}_den.gt",
        altai_out = "results/{m}_model/ibdmix/n100_seed{seed}_altai.ibdmix",
        den_out = "results/{m}_model/ibdmix/n100_seed{seed}_den.ibdmix",
    shell:
        """
        {input.generate_gt} -a {input.altai_vcf} -m {input.modern_vcf} -o {output.altai_gt}
        {input.generate_gt} -a {input.den_vcf} -m {input.modern_vcf} -o {output.den_gt}
        {input.ibdmix} -g {output.altai_gt} --LOD-threshold 4.0 --minor-allele-count-threshold 1 --archaic-error 0.01 --modern-error-max 0.002 --modern-error-proportion 2 -o {output.altai_out}
        {input.ibdmix} -g {output.den_gt} --LOD-threshold 4.0 --minor-allele-count-threshold 1 --archaic-error 0.01 --modern-error-max 0.002 --modern-error-proportion 2 -o {output.den_out}
        """

#######SINGER########

rule get_singer_inputs:
    input:
        singer_base = "singer_base.sh",
        pyscript = "produce_sbatch.py",
    params:
        vcf_dir = "results/{m}_model/msprime",
        relative_path = "results/{m}_model/singer",
        seeds = " ".join([str(x) for x in config["seeds"]]),
    output:
        singer_base = "results/{m}_model/singer/singer_base.sh",
    shell:
        """
        for seed in {params.seeds}
        do
            sed 's|ABSPATHA|'"$(pwd)"'/{params.vcf_dir}|g' {input.singer_base} > {output.singer_base}
            sed 's|ABSPATHB|'"$(pwd)"'/{params.relative_path}|g' {input.singer_base} > {output.singer_base}
            python {input.pyscript} --chunk-size 2000000 --seq-end 50000000 --sbatch-base {output.singer_base} --nposterior 300 --singer-outpref $(pwd)/{params.relative_path}/n100_seed${{seed}} --output {params.relative_path}/sbatch_files/n100_seed${{seed}} --chrom 1
        done
        """


# #######SNPINFO########

# rule get_snpinfo:
#     input:
#         vcf = "results/{m}_model/msprime/n100_seed{seed}.vcf",
#         altai_gt = "results/{m}_model/ibdmix/n100_seed{seed}_altai.gt",
#         den_gt = "results/{m}_model/ibdmix/n100_seed{seed}_den.gt",
#     params:
#         seed = lambda wildcards: int(wildcards.seed),
#     output:
#         snpinfo = "results/{m}_model/snpinfo/n100_seed{seed}.txt",
#     shell:
#         """
#         paste <(bcftools view -M 2 -m 2 {input.vcf} -Ov | bcftools query -f "%CHROM\t%POS\t%REF\t%ALT\n") \
#         <(cut -f5 {input.altai_gt}|tail -n +2) \
#         <(cut -f5 {input.den_gt}|tail -n +2) > {output.snpinfo}
#         echo -e "chrom\tpos\tref\talt\taltai\tdenisova" | cat - {output.snpinfo} > temp && mv temp {output.snpinfo}
#         """

# rule get_singer_mutage:
#     input:
#         trees = expand(
#             "results/{m}_model/singer/n100_seed{seed}_{pp}.tsz",
#             pp = range(250, 300),
#             allow_missing=True,
#         ),
#     output:
#         mutage = expand(
#             "results/{m}_model/singer/n100_seed{seed}_mutage_{pp}.txt",
#             pp=range(250, 300),
#             allow_missing=True,
#         )
#     run:
#         for i in range(len(input.trees)):
#             try:
#                 ts = tszip.decompress(input.trees[i])
#             except Exception as e:
#                 print(f"Error decompressing {input.trees[i]}: {e}")
#                 sys.exit(1)
#             chrom = params.chrom
#             out = ""
#             for tree in ts.trees():
#                 for mut in tree.mutations():
#                     if tree.parent(mut.node) != tskit.NULL:
#                         out += f"{chrom}\t{int(ts.site(mut.site).position) - 1}\t{int(ts.site(mut.site).position)}\t{tree.time(mut.node)}_{tree.time(tree.parent(mut.node))}\n"
#                     else:
#                         out += f"{chrom}\t{int(ts.site(mut.site).position) - 1}\t{int(ts.site(mut.site).position)}\t{tree.time(mut.node)}_{tree.time(mut.node)}\n"
#             out = "chromosome\tposition\tmutation_age\n"
#             for x in a:
#                 out += f"{x.chrom}\t{x.end}\t{x[3]}\n"
#             with open(output.mutage[i], 'w') as f:
#                 f.write(out)

# rule append_af_mutage:
#     input:
#         snpinfo = "results/{m}_model/snpinfo/n100_seed{seed}.txt",
#         singer_mutage = expand(
#             "results/{m}_model/singer/n100_seed{seed}_mutage_{pp}.txt",
#             pp=range(250, 300),
#             allow_missing=True,
#         ),
#         vcf = "results/{m}_model/msprime/n100_seed{seed}.vcf",
#     params:
#         outpref1 = "results/{m}_model/snpinfo/n100_seed{seed}.af",
#         outpref2 = "results/{m}_model/snpinfo/n100_seed{seed}.af.mutage",
#         mutage_pref = "results/{m}_model/singer/n100_seed{seed}_mutage_",
#         snpinfo = "results/{m}_model/snpinfo/n100_seed{seed}",
#         pp_range = range(250, 300),
#         relative_path = "results/{m}_model/snpinfo",
#     output:
#         snpinfo_mutage = "results/{m}_model/snpinfo/n100_seed{seed}.af.mutage.txt",
#     run:
#         afr_sample = f"{params.relative_path}/afr.sample"
#         with open(afr_sample, 'w') as f:
#             for i in range(100, 200):
#                 f.write(f"tsk_{i}\n")
#         SNPINFO().append_AFR_info(params.snpinfo, input.vcf, afr_sample, "AltAF_AFR", params.outpref1)
#         SNPINFO().append_mutage_info(params.outpref1, params.mutage_pref, params.pp_range, params.outpref2)