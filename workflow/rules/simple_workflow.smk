#!python3

import demes
import msprime
import tszip
import numpy as np
import pandas as pd
import yaml
from tqdm import tqdm


rule extract_tmrca_simple:
    """Extract mean and median TMRCAs under ghost admixture in each scenario and push to data frames."""
    input:
        tsz="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.tsz",
        model="results/simulations/models/simple/{model}/{model}_simple_{version}.yaml",
    output:
        null_tmrcas="results/truth/simple/{model}/{version}/n{n}_seed{seed}_tmrcas_null.tsv.gz",
        introgressed_tmrcas="results/truth/simple/{model}/{version}/n{n}_seed{seed}_tmrcas_intro.tsv.gz",
    run:
        ts = tszip.decompress(str(input.tsz))
        # Read in the model and create a population-dictionary
        demo = msprime.Demography.from_demes(demes.load(input.model))
        pop_dict = {i.name: i.id for i in demo.populations}
        arg_utils = ARG_utils(
            afr_size=int(wildcards.n),
            eur_size=0,
            afr_poplabel=pop_dict["A"],
            ghost_poplabel=pop_dict["B"],
            human_ancestor_poplabel=pop_dict["ANC"],
        )
        arg_utils.add_tree_sequence(ts)
        common, afr, null = arg_utils.extract_tmrca_all()
        pd.DataFrame.from_dict(
            {"Mean_TMRCA_Null": null[0], "Median_TMRCA_Null": null[2]}
        ).to_csv(output.null_tmrcas, sep="\t", index=None)
        pd.DataFrame.from_dict(
            {"Mean_TMRCA_Intro": afr[0], "Median_TMRCA_Intro": afr[2]}
        ).to_csv(output.introgressed_tmrcas, sep="\t", index=None)


rule extract_branch_bound_simple:
    """Extract branch boundaries under ghost admixture in each scenario and push to data frames."""
    input:
        tsz="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.tsz",
        model="results/simulations/models/simple/{model}/{model}_simple_{version}.yaml",
    output:
        null_branches="results/truth/simple/{model}/{version}/n{n}_seed{seed}_branch_null.tsv.gz",
        intro_branches="results/truth/simple/{model}/{version}/n{n}_seed{seed}_branch_intro.tsv.gz",
        all_branches="results/simulations/simple/{model}/{version}/n{n}_seed{seed}_empneed.npz",
    params:
        t=lambda wildcards: simple_config[wildcards.model]["versions"][
            wildcards.version
        ]["migration_time"]
        / 29,
        t_arch=lambda wildcards: simple_config[wildcards.model]["versions"][
            wildcards.version
        ]["archaic_split"]
        / 29,
    run:
        ts = tszip.decompress(str(input.tsz))
        demo = msprime.Demography.from_demes(demes.load(input.model))
        pop_dict = {i.name: i.id for i in demo.populations}
        arg_utils = ARG_utils(
            afr_size=int(wildcards.n),
            eur_size=0,
            afr_poplabel=pop_dict["A"],
            ghost_poplabel=pop_dict["B"],
            human_ancestor_poplabel=pop_dict["ANC"],
        )
        arg_utils.add_tree_sequence(ts)
        # NOTE: this is still in generations ...
        _, upper_intro_all, _, upper_null_all = arg_utils.extract_branch_boundaries_all(
            cond=params.t
        )
        pd.DataFrame.from_dict({"Upper_Branch_Null": upper_null_all}).to_csv(
            output.null_branches, sep="\t", index=None
        )
        pd.DataFrame.from_dict({"Upper_Branch_Intro": upper_intro_all}).to_csv(
            output.intro_branches, sep="\t", index=None
        )
        # split at 750000ya, 25862gen, here to extract params need in empirical filtering
        hmm = ARG_HMM_BRANCH(t_admix=int(params.t))
        hmm.add_tree_sequence(ts)
        hmm.extract_branch_boundaries_bulk(
            nsamp=int(wildcards.n), population=pop_dict["A"]
        )
        np.savez_compressed(
            str(output.all_branches),
            upper_bound=hmm.bulk_t_upper,
            nodes=hmm.bulk_intro_node,
            t_admix=int(params.t),
            t_archaic=params.t_arch,
        )


rule infer_gamma_mix_tmrca_simple:
    """Infer the gamma mixture distribution from the true tmrcas.

    NOTE: this is not under the HMM model.
    NOTE: this is actually requiring the usage of the true tmrcas from null and introgressed categories.
    """
    input:
        null_tmrcas="results/truth/simple/{model}/{version}/n{n}_seed{seed}_tmrcas_null.tsv.gz",
        introgressed_tmrcas="results/truth/simple/{model}/{version}/n{n}_seed{seed}_tmrcas_intro.tsv.gz",
    output:
        est_params="results/simulations/simple/{model}/{version}/params/n{n}_seed{seed}.tmrcas.est_true.yaml",
    run:
        null_tmrcas_df = pd.read_csv(input.null_tmrcas, sep="\t")
        intro_tmrcas_df = pd.read_csv(input.introgressed_tmrcas, sep="\t")
        hmm = ARG_HMM_BRANCH()
        null_tmrcas = null_tmrcas_df.Median_TMRCA_Null.values
        intro_tmrcas = intro_tmrcas_df.Median_TMRCA_Intro.values
        a1, b1 = hmm.est_gamma_mle(null_tmrcas)
        a2, b2 = hmm.est_gamma_mle(intro_tmrcas)
        w_est = intro_tmrcas.size / (null_tmrcas.size + intro_tmrcas.size)
        param_dict = {
            "a1": float(a1),
            "b1": float(b1),
            "a2": float(a2),
            "b2": float(b2),
            "w": float(w_est),
        }
        with open(output.est_params, "w") as out:
            yaml.dump(param_dict, out, default_flow_style=False)


rule infer_gamma_mix_branch_bound_simple:
    """Infer the gamma mixture distribution from the true branch lengths.

    NOTE: this is not under the HMM model.
    NOTE: this is actually requiring the usage of the true branch-lengths from null and introgressed categories.
    """
    input:
        null_branches="results/truth/simple/{model}/{version}/n{n}_seed{seed}_branch_null.tsv.gz",
        intro_branches="results/truth/simple/{model}/{version}/n{n}_seed{seed}_branch_intro.tsv.gz",
    output:
        est_params="results/simulations/simple/{model}/{version}/params/n{n}_seed{seed}.branch_length.est_true.yaml",
    run:
        null_branches_df = pd.read_csv(input.null_branches, sep="\t")
        intro_branches_df = pd.read_csv(input.intro_branches, sep="\t")
        hmm = ARG_HMM_BRANCH()
        null_branches = null_branches_df.Upper_Branch_Null.values
        intro_branches = intro_branches_df.Upper_Branch_Intro.values
        a1, b1 = hmm.est_gamma_mle(null_branches)
        a2, b2 = hmm.est_gamma_mle(intro_branches)
        w_est = intro_branches.size / (null_branches.size + intro_branches.size)
        param_dict = {
            "a1": float(a1),
            "b1": float(b1),
            "a2": float(a2),
            "b2": float(b2),
            "w": float(w_est),
        }
        with open(output.est_params, "w") as out:
            yaml.dump(param_dict, out, default_flow_style=False)


rule extract_branch_stats_tsz_simple:
    """Extract tmrcas/branch lengths for simple plug in."""
    input:
        tsz="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.tsz",
        model="results/simulations/models/simple/{model}/{model}_simple_{version}.yaml",
    output:
        txs_data="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.{stat}.npz",
    wildcard_constraints:
        stat="tmrcas|branch_length",
    params:
        t=lambda wildcards: simple_config[wildcards.model]["versions"][
            wildcards.version
        ]["migration_time"]
        / 29,
    run:
        n = int(wildcards.n)
        t = params.t
        ts = tszip.decompress(str(input.tsz))
        demo = msprime.Demography.from_demes(demes.load(input.model))
        pop_dict = {i.name: i.id for i in demo.populations}
        arg_utils = ARG_utils(
            afr_size=n,
            eur_size=0,
            afr_poplabel=pop_dict["A"],
            ghost_poplabel=pop_dict["B"],
            human_ancestor_poplabel=pop_dict["ANC"],
        )
        arg_utils.add_tree_sequence(ts)
        pos = np.zeros(ts.num_trees)
        txs = np.zeros(shape=(n, ts.num_trees))
        for j, tree in tqdm(enumerate(arg_utils.ts.trees())):
            pos[j] = np.mean(tree.interval)
            for i in range(n):
                if wildcards.stat == "tmrcas":
                    mean_t, _, med_t, _ = arg_utils.extract_tmrca(
                        [i], arg_utils.afr_samples, tree
                    )
                    txs[i, j] = med_t[0]
                elif wildcards.stat == "branch_length":
                    longest, lower_branch, upper_branch = arg_utils.get_longest_branch(
                        [i], t, tree
                    )
                    if upper_branch[0] <= t:
                        txs[i, j] = t + 1e-5
                    else:
                        txs[i, j] = upper_branch[0]
        np.savez_compressed(str(output.txs_data), txs=txs, pos=pos)


rule infer_params_hmm_tsv:
    """Infer parameters under the branch-bound HMM (but with the tmrca_setting)

    NOTE: the data must be named hmm_data for the script to work.
    """
    input:
        hmm_data="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.{stat}.npz",
    output:
        inf_hmm_params="results/simulations/simple/{model}/{version}/params/n{n}_seed{seed}.{stat}.hmm_est.tsv",
    wildcard_constraints:
        stat="tmrcas|branch_length",
    params:
        niter=20,
        t=lambda wildcards: simple_config[wildcards.model]["versions"][
            wildcards.version
        ]["migration_time"]
        / 29,
        mode=lambda wildcards: "tmrca" if wildcards.stat == "tmrcas" else "branch",
    script:
        "../scripts/infer_hmm_params.py"


rule infer_params_hmm_yaml:
    input:
        inf_hmm_params_tsv="results/simulations/simple/{model}/{version}/params/n{n}_seed{seed}.{stat}.hmm_est.tsv",
    output:
        inf_hmm_params_yaml="results/simulations/simple/{model}/{version}/params/n{n}_seed{seed}.{stat}.hmm_est.yaml",
    wildcard_constraints:
        stat="tmrcas|branch_length",
    run:
        hmm_params_df = pd.read_csv(input.inf_hmm_params_tsv, sep="\t")
        final_params = hmm_params_df.iloc[-1, :]
        est_params = {
            "a1": float(final_params.alpha_null),
            "b1": float(final_params.beta_null),
            "a2": float(final_params.alpha_intro),
            "b2": float(final_params.beta_intro),
            "w": float(final_params.w),
            "p": float(final_params.p),
            "q": float(final_params.q),
        }
        with open(output.inf_hmm_params_yaml, "w") as out:
            yaml.dump(est_params, out, default_flow_style=False)


rule infer_params_gamma_mixture_yaml:
    input:
        hmm_data="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.{stat}.npz",
    output:
        inf_gamma_params="results/simulations/simple/{model}/{version}/params/n{n}_seed{seed}.{stat}.mixture_gamma_est.yaml",
    wildcard_constraints:
        stat="tmrcas|branch_length",
    threads: 4
    params:
        t=lambda wildcards: 0
        if wildcards.stat == "tmrcas"
        else simple_config[wildcards.model]["versions"][wildcards.version][
            "migration_time"
        ]
        / 29,
    script:
        "../scripts/infer_gamma_params.py"


rule est_binary_emissions:
    """Estimate a binary emission model based on t_archaic."""
    input:
        hmm_data="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.{stat}.npz",
    output:
        inf_binary_emission="results/simulations/simple/{model}/{version}/params/n{n}_seed{seed}.{stat}.binary_est.yaml",
    wildcard_constraints:
        stat="tmrcas|branch_length",
    threads: 2
    params:
        t=lambda wildcards: simple_config[wildcards.model]["versions"][
            wildcards.version
        ]["migration_time"]
        / 29,
        t_archaic=lambda wildcards: simple_config[wildcards.model]["versions"][
            wildcards.version
        ]["archaic_split"]
        / 29,
    run:
        hmm = ARG_HMM_BRANCH()
        data = np.load(input["hmm_data"])  # noqa
        txs_sim = data["txs"]
        hmm.t_admix = params.t
        hmm.bulk_t_upper = txs_sim
        hmm.set_emission_binning(mode="bulk")
        t_null_prob, t_admix_prob, t_archaic = hmm.emission_dist_binary(
            txs=txs_sim, t_archaic=params.t_archaic, eps=1e-8
        )
        param_dict = {
            "a1": 1.0,
            "b1": 1.0,
            "a2": 1.0,
            "b2": 1.0,
            "w": float(t_admix_prob),
            "p": 0.01,
            "q": 0.1,
            "t_null_prob": float(t_null_prob),
            "t_admix_prob": float(t_admix_prob),
            "t_archaic": float(t_archaic),
        }
        with open(output.inf_binary_emission, "w") as out:
            yaml.dump(param_dict, out, default_flow_style=False)


rule run_arg_hmm_forward_backward_simple:
    """Given certain parameters inferred, evaluate the posterior probability of introgression based on TMRCAs"""
    input:
        hmm_data="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.{stat}.npz",
        hmm_params="results/simulations/simple/{model}/{version}/params/n{n}_seed{seed}.{stat}.{est_method}.yaml",
    output:
        posterior_data="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.{stat}.{est_method}.npz",
    wildcard_constraints:
        stat="tmrcas|branch_length",
        est_methods="est_true|hmm_est|mixture_gamma_est|binary_est",
    params:
        t=lambda wildcards: 0
        if wildcards.stat == "tmrcas"
        else simple_config[wildcards.model]["versions"][wildcards.version][
            "migration_time"
        ]
        / 29,
    script:
        "../scripts/infer_hmm_posterior.py"


rule evaluate_hmm_performance:
    """Rule to evaluate precision and recall across hmm results."""
    input:
        hmm_posterior="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.{stat}.{est_method}.npz",
        emp_data="results/simulations/simple/{model}/{version}/n{n}_seed{seed}_empneed.npz",
        tsz="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.tsz",
        ind_bed="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.indiv.bed",
        merged_bed="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.merged.bed",
    output:
        performance_stats="results/simulations/simple/{model}/{version}/n{n}_seed{seed}.{stat}.{est_method}.performance_hmm.tsv",
    wildcard_constraints:
        stat="tmrcas|branch_length",
        est_methods="est_true|hmm_est|mixture_gamma_est|binary_est",
    params:
        mode="hmm",
        n=lambda wildcards: int(wildcards.n),
        t=lambda wildcards: 0
        if wildcards.stat == "tmrcas"
        else simple_config[wildcards.model]["versions"][wildcards.version][
            "migration_time"
        ]
        / 29,
    script:
        "../scripts/eval_performance.py"
