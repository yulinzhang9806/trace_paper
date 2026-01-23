#!python3

import numpy as np
import tskit
import tszip
import msprime
import sys
import os
from workflow.scripts.utils import Analysis_workflow_utils
from utils import Performance_utils
import demes

rule general_indiv:
    """Run TRACE."""
    input:
        target_tsz=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_full.tsz",
        target_sub_tsz=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_A.tsz",
        outgroup_tsz=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_null.tsz",
        outgroup_sub_tsz=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_null_A.tsz",
    output:
        ind_output=str(paths["benchmark_trace_outputs"]) + "/{testing_scene}/{model}/{version}/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
        ind_output_sub=str(paths["benchmark_trace_outputs"]) + "/{testing_scene}/{model}/{version}/n{n}_seed{seed}_{t}_{intro}_A_ppsind{ind}.npz",
    wildcard_constraints:
        ind = "|".join([str(i) for i in INDS])  # explicitly define possible values
        STEP = "|".join(["error_input", "extreme_demo", "inference_range"])  # explicitly define possible values
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        ind=lambda wildcards: wildcards.ind,
        seed=lambda wildcards: wildcards.seed,
        version=lambda wildcards: wildcards.version,
        intro_prop=lambda wildcards: wildcards.intro,
        pp_outpref=str(paths["benchmark_trace_outputs"]) + "/{testing_scene}/{model}/{version}/n{n}_seed{seed}_{t}_{intro}_pps",
        pp_outpref_sub=str(paths["benchmark_trace_outputs"]) + "/{testing_scene}/{model}/{version}/n{n}_seed{seed}_{t}_{intro}_A_pps",
    threads: 2
    run:
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_hmm_results_ondind(
            ind=int(params.ind),
            target_tsz = input.target_tsz,
            outgroup_tsz = input.outgroup_tsz,
            pp_output = params.pp_outpref,
            t_archaic = int(params.t),
            intro_prop = float(params.intro_prop),
            seed = int(params.seed),
        )
        analysis_utils.record_hmm_results_ondind(
            ind=int(params.ind),
            target_tsz = input.target_sub_tsz,
            outgroup_tsz = input.outgroup_sub_tsz,
            pp_output = params.pp_outpref_sub,
            t_archaic = int(params.t),
            intro_prop = float(params.intro_prop),
            seed = int(params.seed),
        )
        print("Messages", file=sys.stderr)

rule general_eval:
    """Evaluate TRACE outputs."""
    input:
        expand(
            str(paths["benchmark_trace_outputs"]) + "/{testing_scene}/{model}/{version}/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
        expand(
            str(paths["benchmark_trace_outputs"]) + "/{testing_scene}/{model}/{version}/n{n}_seed{seed}_{t}_{intro}_A_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
        ind_bed=ancient(str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}.indiv.bed"),
    output:
        perf_output=str(paths["benchmark_trace_outputs"]) + "/{testing_scene}/{model}/{version}/n{n}_seed{seed}_{t}_{intro}_data.tsv",
        perf_output_sub=str(paths["benchmark_trace_outputs"]) + "/{testing_scene}/{model}/{version}/n{n}_seed{seed}_{t}_{intro}_A_data.tsv",
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        l_cutoff=lambda wildcards: analysis_input["l_cutoff"],
        pp_cutoff=lambda wildcards: analysis_input["pp_cutoff"],
        inds=lambda wildcards: analysis_input["{testing_scene}"]["inds"],
        sequence_length=50e6,
        seed=lambda wildcards: wildcards.seed,
        version=lambda wildcards: wildcards.version,
        intro_prop=lambda wildcards: wildcards.intro,
        pp_outpref=str(paths["benchmark_trace_outputs"]) + "/{testing_scene}/{model}/{version}/n{n}_seed{seed}_{t}_{intro}_ppsind",
        pp_outpref_sub=str(paths["benchmark_trace_outputs"]) + "/{testing_scene}/{model}/{version}/n{n}_seed{seed}_{t}_{intro}_A_ppsind",
    run:
        intro_prop = float(params.intro_prop)
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_performance(
            inds=int(params.inds),
            ind_bed=input.ind_bed,
            n=params.n,
            pp_outpref=params.pp_outpref,
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            sequence_length=params.sequence_length,
            intro_prop=intro_prop,
            perf_output=output.perf_output,
            maxlen=None,
        )
        analysis_utils.record_performance(
            inds=int(params.inds),
            ind_bed=input.ind_bed,
            n=params.n,
            pp_outpref=params.pp_outpref_sub,
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            sequence_length=params.sequence_length,
            intro_prop=intro_prop,
            perf_output=output.perf_output_sub,
            maxlen=None,
        )

rule inference_comparison_TRACE_indiv:
    """Run TRACE."""
    input:
        target_tsz=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_A_full.tsz",
        outgroup_tsz=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC_null_full.tsz",
        yaml=str(paths["benchmark_simulations"]) + "/models/{model}/{model}_{version}.yaml",
    output:
        ind_output=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/TRACE/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
    wildcard_constraints:
        ind = "|".join([str(i) for i in INDS])  # explicitly define possible values
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        ind=lambda wildcards: wildcards.ind,
        seed=lambda wildcards: wildcards.seed,
        version=lambda wildcards: wildcards.version,
        intro_prop=None,
        pp_outpref=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/TRACE/n{n}_seed{seed}_{t}_{intro}_pps",
    threads: 2
    run:
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_hmm_results_ondind(
            ind=int(params.ind),
            target_tsz = input.target_tsz,
            outgroup_tsz = input.outgroup_tsz,
            pp_output = params.pp_outpref,
            t_archaic = int(params.t),
            intro_prop = params.intro_prop,
            seed = int(params.seed),
        )
        print("Messages", file=sys.stderr)

rule inference_comparison_TRACE_eval:
    """Evaluate TRACE outputs."""
    input:
        expand(
            str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/TRACE/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
        ind_bed=ancient(str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.indiv.bed"),
    output:
        perf_output=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/TRACE/n{n}_seed{seed}_{t}_{intro}_data.tsv",
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        l_cutoff=lambda wildcards: analysis_input["l_cutoff"],
        pp_cutoff=lambda wildcards: analysis_input["pp_cutoff"],
        inds=lambda wildcards: analysis_input["inference_comparison"]["inds"],
        sequence_length=250e6,
        seed=lambda wildcards: wildcards.seed,
        version=lambda wildcards: wildcards.version,
        intro_prop=lambda wildcards: wildcards.intro,
        pp_outpref=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/TRACE/n{n}_seed{seed}_{t}_{intro}_ppsind",
    run:
        intro_prop = float(params.intro_prop)
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_performance(
            inds=int(params.inds),
            ind_bed=input.ind_bed,
            n=params.n,
            pp_outpref=params.pp_outpref,
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            sequence_length=params.sequence_length,
            intro_prop=intro_prop,
            perf_output=output.perf_output,
            maxlen=None,
        )

rule inference_comparison_hmmix:
    """Generate data for comparing performance of HMM with hmmix."""
    input:
        expand(
            str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_{ind}.hap1.txt",
            ind=range(5),
            allow_missing=True,
        ),
        expand(
            str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_{ind}.hap2.txt",
            ind=range(5),
            allow_missing=True,
        ),
        expand(
            str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_{ind}.hap1.txt",
            ind=range(5),
            allow_missing=True,
        ),
        expand(
            str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_{ind}.hap2.txt",
            ind=range(5),
            allow_missing=True,
        ),
        ind_bed=ancient(str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.indiv.bed"),
    output:
        perf_output=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_{t}_{intro}_data.tsv",
    params:
        n=lambda wildcards: wildcards.n,
        inds=lambda wildcards: analysis_input["inference_comparison"]["inds"],
        pref=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}.tsk_",
        pref_null=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/hmmix/n{n}_seed{seed}_null.tsk_",
        intro=lambda wildcards: wildcards.intro,
        l_cutoff=lambda wildcards: analysis_input["l_cutoff"],
        pp_cutoff=lambda wildcards: analysis_input["pp_cutoff"],
        version=lambda wildcards: wildcards.version,
    run:
        intro_prop = float(params.intro)
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.hmmix_performance(
            inds=int(params.inds),
            inputpref=str(params.pref),
            inputpref_null=str(params.pref_null),
            exp_intro=intro_prop*250,
            ind_bed=str(input.ind_bed),
            n=int(params.n),
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            perf_output=str(output.perf_output),
        )
        print("Messages", file=sys.stderr)

rule inference_comparison_ibdmix:
    """Generate data for comparing performance of HMM with ibdmix."""
    input:
        Aout = str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/ibdmix/n{n}_seed{seed}_A.ibdmix",
        nullout = str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/ibdmix/n{n}_seed{seed}_null.ibdmix",
        ind_bed=ancient(str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.indiv.bed"),
    output:
        perf_output=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/ibdmix/n{n}_seed{seed}_{t}_{intro}_data.tsv",
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        inds=lambda wildcards: analysis_input["inference_comparison"]["inds"],
        intro=lambda wildcards: wildcards.intro,
        l_cutoff=lambda wildcards: analysis_input["l_cutoff"],
        pp_cutoff=lambda wildcards: analysis_input["pp_cutoff"],
        slod=lambda wildcards: analysis_input["inference_comparison"]["slod"],
        version=lambda wildcards: wildcards.version,
    run:
        intro_prop = float(params.intro)
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.ibdmix_performance(
            inds=int(params.inds),
            inputf=str(input.Aout),
            inputf_null=str(input.nullout),
            exp_intro=intro_prop*250,
            ind_bed=str(input.ind_bed),
            n=int(params.n),
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            slod_cutoff=params.slod,
            perf_output=str(output.perf_output),
        )
        print("Messages", file=sys.stderr)

rule inference_comparison_sprime:
    """Generate data for comparing performance of HMM with sprime."""
    input:
        scorefile=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_Sprime.score",
        scorefile_null=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_null_Sprime.score",
        ind_bed=ancient(str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/simulations/n{n}_seed{seed}_withARC.indiv.bed"),
    output:
        perf_output=str(paths["benchmark_trace_outputs"]) + "/inference_comparison/{model}/{version}/sprime/n{n}_seed{seed}_{t}_{intro}_data.tsv",
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        inds=lambda wildcards: analysis_input["inference_comparison"]["inds"],
        score=lambda wildcards: analysis_input["inference_comparison"]["score"],
        l_cutoff=lambda wildcards: analysis_input["l_cutoff"],
        pp_cutoff=lambda wildcards: analysis_input["pp_cutoff"],
        version=lambda wildcards: wildcards.version,
        intro=lambda wildcards: wildcards.intro,
    run:
        exp_intro = float(params.intro) * 250
        out_perf = {
            "individual": [],
            "mean_posterior_cutoff": [],
            "length_cutoff": [],
            "precision": [],
            "recall": [],
            "false_discovery": [],
        }
        analysis_utils = Analysis_workflow_utils()
        truth = Performance_utils().read_truth_bed(input.ind_bed, 2 * int(params.n))
        t = truth[0]
        for i in range(2 * int(inds)):
            if len(t) == 0:
                t = truth[i]
            else:
                if len(truth[i]) > 0:
                    t = np.append(t, truth[i], axis=0)
        for s in range(len(params.score)):
            for l in params.l_cutoff:
                pre, rec, fdc = analysis_utils.calculate_sprime(
                    input.scorefile, input.scorefile_null, t, min_l=l, score=params.score[s]
                )
                out_perf["individual"].append(1)
                out_perf["mean_posterior_cutoff"].append(params.pp_cutoff[s])
                out_perf["length_cutoff"].append(l)
                out_perf["precision"].append(pre)
                out_perf["recall"].append(rec)
                out_perf["false_discovery"].append(fdc / exp_intro)
        out_pref = pd.DataFrame.from_dict(out_perf)
        out_pref.to_csv(output.perf_output, sep="\t", index=False)
        print("Messages", file=sys.stderr)

rule arg_infer_msprime_indiv:
    input:
        target_tsz=ancient(str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_full.tsz"),
        target_sub_tsz=ancient(str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_A.tsz"),
        outgroup_tsz=ancient(str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_null.tsz"),
        outgroup_sub_tsz=ancient(str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}_null_A.tsz"),
    output:
        ind_output=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/msprime/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
        ind_output_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/msprime/n{n}_seed{seed}_{t}_{intro}_A_ppsind{ind}.npz",
    wildcard_constraints:
        ind = "|".join([str(i) for i in INDS])  # explicitly define possible values
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        ind=lambda wildcards: wildcards.ind,
        seed=lambda wildcards: wildcards.seed,
        version=lambda wildcards: wildcards.version,
        intro=None,
        pp_outpref=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/msprime/n{n}_seed{seed}_{t}_{intro}_pps",
        pp_outpref_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/msprime/n{n}_seed{seed}_{t}_{intro}_A_pps",
    threads: 2
    run:
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_hmm_results_ondind(
            ind=int(params.ind),
            target_tsz = input.target_tsz,
            outgroup_tsz = input.outgroup_tsz,
            pp_output = params.pp_outpref,
            t_archaic = int(params.t),
            intro_prop = params.intro,
            seed = int(params.seed),
        )
        analysis_utils.record_hmm_results_ondind(
            ind=int(params.ind),
            target_tsz = input.target_sub_tsz,
            outgroup_tsz = input.outgroup_sub_tsz,
            pp_output = params.pp_outpref_sub,
            t_archaic = int(params.t),
            intro_prop = params.intro,
            seed = int(params.seed),
        )
        print("Messages", file=sys.stderr)

rule arg_infer_msprime_eval:
    input:
        expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/msprime/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
        expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/msprime/n{n}_seed{seed}_{t}_{intro}_A_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
        ind_bed=ancient(str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}.indiv.bed"),
    output:
        perf_output=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/msprime/n{n}_seed{seed}_{t}_{intro}_data.tsv",
        perf_output_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/msprime/n{n}_seed{seed}_{t}_{intro}_A_data.tsv",
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        l_cutoff=lambda wildcards: analysis_input["l_cutoff"],
        pp_cutoff=lambda wildcards: analysis_input["pp_cutoff"],
        inds=lambda wildcards: analysis_input["error_input"]["inds"],
        sequence_length=50e6,
        seed=lambda wildcards: wildcards.seed,
        version=lambda wildcards: wildcards.version,
        intro=lambda wildcards: wildcards.intro,
        pp_outpref=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/msprime/n{n}_seed{seed}_{t}_{intro}_ppsind",
        pp_outpref_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/msprime/n{n}_seed{seed}_{t}_{intro}_A_ppsind",
    run:
        intro_prop = float(params.intro)
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_performance(
            inds=int(params.inds),
            ind_bed=input.ind_bed,
            n=params.n,
            pp_outpref=params.pp_outpref,
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            sequence_length=params.sequence_length,
            intro_prop=intro_prop,
            perf_output=output.perf_output,
            maxlen=None,
        )
        analysis_utils.record_performance(
            inds=int(params.inds),
            ind_bed=input.ind_bed,
            n=params.n,
            pp_outpref=params.pp_outpref_sub,
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            sequence_length=params.sequence_length,
            intro_prop=intro_prop,
            perf_output=output.perf_output_sub,
            maxlen=None,
        )

rule arg_infer_relate_indiv:
    """Generate data for performance of HMM on relate trees."""
    input:
        target_tsz=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A_full.trees",
        target_sub_tsz=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_A.trees",
        outgroup_tsz=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out_full.trees",
        outgroup_sub_tsz=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_out.trees",
        yaml=ancient(str(paths["benchmark_simulations"]) + "/models/{model}/{model}_{version}.yaml"),
    output:
        ind_output=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
        ind_output_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_{t}_{intro}_A_ppsind{ind}.npz",
    wildcard_constraints:
        ind = "|".join([str(i) for i in INDS])  # explicitly define possible values
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        ind=lambda wildcards: wildcards.ind,
        seed=lambda wildcards: wildcards.seed,
        version=lambda wildcards: wildcards.version,
        intro=None,
        pp_outpref=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_{t}_{intro}_pps",
        pp_outpref_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_{t}_{intro}_A_pps",
    threads: 2
    run:
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_hmm_results_ondind(
            ind=int(params.ind),
            target_tsz = input.target_tsz,
            outgroup_tsz = input.outgroup_tsz,
            pp_output = params.pp_outpref,
            t_archaic = int(params.t),
            intro_prop = params.intro,
            seed = int(params.seed),
        )
        analysis_utils.record_hmm_results_ondind(
            ind=int(params.ind),
            target_tsz = input.target_sub_tsz,
            outgroup_tsz = input.outgroup_sub_tsz,
            pp_output = params.pp_outpref_sub,
            t_archaic = int(params.t),
            intro_prop = params.intro,
            seed = int(params.seed),
        )
        print("Messages", file=sys.stderr)

rule arg_infer_relate_eval:
    """Evaluate performance of GhostHMM."""
    input:
        expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
        expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_{t}_{intro}_A_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
        ind_bed=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}.indiv.bed",
    output:
        perf_output=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_{t}_{intro}_data.tsv",
        perf_output_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_{t}_{intro}_A_data.tsv",
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        l_cutoff=lambda wildcards: analysis_input["l_cutoff"],
        pp_cutoff=lambda wildcards: analysis_input["pp_cutoff"],
        inds=lambda wildcards: analysis_input["error_input"]["inds"],
        sequence_length=50e6,
        seed=lambda wildcards: wildcards.seed,
        version=lambda wildcards: wildcards.version,
        intro=lambda wildcards: wildcards.intro,
        pp_outpref=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_{t}_{intro}_ppsind",
        pp_outpref_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/relate/n{n}_seed{seed}_{t}_{intro}_A_ppsind",
    run:
        intro_prop = float(params.intro)
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_performance(
            inds=int(params.inds),
            ind_bed=input.ind_bed,
            n=params.n,
            pp_outpref=params.pp_outpref,
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            sequence_length=params.sequence_length,
            intro_prop=intro_prop,
            perf_output=output.perf_output,
            maxlen=None,
        )
        analysis_utils.record_performance(
            inds=int(params.inds),
            ind_bed=input.ind_bed,
            n=params.n,
            pp_outpref=params.pp_outpref_sub,
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            sequence_length=params.sequence_length,
            intro_prop=intro_prop,
            perf_output=output.perf_output_sub,
            maxlen=None,
        )

rule arg_infer_singer_record:
    """Generate data for running HMM on singer trees."""
    input:
        target_tsz=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_full_{STEP}.tsz",
        target_sub_tsz=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_{STEP}.tsz",
        outgroup_tsz=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_full_{STEP}.tsz",
        outgroup_sub_tsz=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_{STEP}.tsz",
    output:
        pt_output=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_ppsts{STEP}.npz",
        pt_output_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_ppsts{STEP}.npz",
    wildcard_constraints:
        STEP = "|".join([str(i) for i in range(150, 200)])  # explicitly define possible values
    params:
        ind = np.array(INDS),
        step = lambda wildcards: wildcards.STEP,
        func = "mean",
        windowsize = 1000,
        t = lambda wildcards: wildcards.t,
        target_filepath=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_full_",
        target_sub_filepath=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_A_",
        outgroup_filepath=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_full_",
        outgroup_sub_filepath=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_out_",
        pt_outpref=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_ppsts",
        pt_outpref_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_ppsts",
    run:
        if params.func == "median":
            func = np.median
        elif params.func == "mean":
            func = np.mean
        else:
            raise ValueError("Function not recognized. Use 'mean' or 'median'.")
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_hmm_data_singer_singlets(
            ind = params.ind,
            targetpref = params.target_filepath,
            outgrouppref = params.outgroup_filepath,
            asamp = params.step,
            windowsize = params.windowsize,
            t_archaic = int(params.t),
            pp_output = params.pt_outpref,
            func = func,
        )
        analysis_utils.record_hmm_data_singer_singlets(
            ind = params.ind,
            targetpref = params.target_sub_filepath,
            outgrouppref = params.outgroup_sub_filepath,
            asamp = params.step,
            windowsize = params.windowsize,
            t_archaic = int(params.t),
            pp_output = params.pt_outpref,
            func = func,
        )
        print("Messages", file=sys.stderr)

rule arg_infer_singer_indiv:
    """Generate data for performance of HMM on singer trees."""
    input:
        expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_ppsts{step}.npz",
            step=range(150, 200), allow_missing=True,
        ),
        expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_ppsts{step}.npz",
            step=range(150, 200), allow_missing=True,
        ),
    output:
        ind_output=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
        ind_output_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_ppsind{ind}.npz",
    wildcard_constraints:
        ind = "|".join([str(i) for i in INDS])  # explicitly define possible values
    params:
        t=lambda wildcards: int(wildcards.t),
        ind=lambda wildcards: int(wildcards.ind),
        seed=lambda wildcards: int(wildcards.seed),
        intro=None,
        func = "mean",
        full_filepath=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_ppsts",
        sub_filepath=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_ppsts",
        pp_outpref=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_pps",
        pp_outpref_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_pps",
    run:
        if params.func == "median":
            func = np.median
        elif params.func == "mean":
            func = np.mean
        else:
            raise ValueError("Function not recognized. Use 'mean' or 'median'.")
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_hmm_results_avg_oneind(
            ind=int(params.ind),
            npz_filepath = params.full_filepath,
            fsample = range(150, 200),
            pp_output = params.pp_outpref,
            t_archaic = int(params.t),
            intro_prop = params.intro,
            func = func,
            seed = int(params.seed),
        )
        analysis_utils.record_hmm_results_avg_oneind(
            ind=int(params.ind),
            npz_filepath = params.sub_filepath,
            fsample = range(150, 200),
            pp_output = params.pp_outpref_sub,
            t_archaic = int(params.t),
            intro_prop = params.intro,
            func = func,
            seed = int(params.seed),
        )
        print("Messages", file=sys.stderr)

rule arg_infer_singer_eval:
    """Evaluate performance of GhostHMM."""
    input:
        expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
        expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
        ind_bed=str(paths["benchmark_simulations"]) + "/outputs/{model}/{version}/n{n}_seed{seed}.indiv.bed",
    output:
        perf_output=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_data.tsv",
        perf_output_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_data.tsv",
    params:
        n=lambda wildcards: wildcards.n,
        t=lambda wildcards: wildcards.t,
        l_cutoff=lambda wildcards: analysis_input["l_cutoff"],
        pp_cutoff=lambda wildcards: analysis_input["pp_cutoff"],
        inds=lambda wildcards: analysis_input["error_input"]["inds"],
        sequence_length=50e6,
        seed=lambda wildcards: wildcards.seed,
        version=lambda wildcards: wildcards.version,
        intro=lambda wildcards: wildcards.intro,
        pp_outpref=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_ppsind",
        pp_outpref_sub=str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_ppsind",
    run:
        intro_prop = float(params.intro)
        analysis_utils = Analysis_workflow_utils()
        analysis_utils.record_performance(
            inds=int(params.inds),
            ind_bed=input.ind_bed,
            n=params.n,
            pp_outpref=params.pp_outpref,
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            sequence_length=params.sequence_length,
            intro_prop=intro_prop,
            perf_output=output.perf_output,
            maxlen=None,
        )
        analysis_utils.record_performance(
            inds=int(params.inds),
            ind_bed=input.ind_bed,
            n=params.n,
            pp_outpref=params.pp_outpref_sub,
            l_cutoff=params.l_cutoff,
            pp_cutoff=params.pp_cutoff,
            sequence_length=params.sequence_length,
            intro_prop=intro_prop,
            perf_output=output.perf_output_sub,
            maxlen=None,
        )

rule arg_infer_singer_cleanup:
    """Cleanup intermediate files from singer trees."""
    input:
        full_npz = expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_ppsts{step}.npz",
            step=range(150, 200),
            allow_missing=True,
        ),
        sub_npz = expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_ppsts{step}.npz",
            step=range(90, 100),
            allow_missing=True,
        ),
        ind_npz = expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
        ind_sub_npz = expand(
            str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_A_ppsind{ind}.npz",
            ind=INDS, allow_missing=True,
        ),
    output:
        log = str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_cleanup.log",
    params:
        cleanpath = str(paths["benchmark_trace_outputs"]) + "/arg_infer/{model}/{version}/singer/n{n}_seed{seed}_{t}_{intro}_"
    shell:
        """
        rm {params.cleanpath}*ppsts*npz
        echo "Cleanup completed" > {output.log}
        """

