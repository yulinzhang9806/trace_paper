#!python3


rule simulate_hmm_data:
    """Simulate data under the double gamma emission model."""
    output:
        hmm_data="results/simulations/hmm_joint_gamma/{sim}.rep{seed}.npz",
    wildcard_constraints:
        seed="\d+",
    params:
        n=lambda wildcards: sim_config[wildcards.sim]["n"],
        l=lambda wildcards: sim_config[wildcards.sim]["l"],
        t=lambda wildcards: sim_config[wildcards.sim]["t"],
        a1=lambda wildcards: sim_config[wildcards.sim]["a1"],
        b1=lambda wildcards: sim_config[wildcards.sim]["b1"],
        a2=lambda wildcards: sim_config[wildcards.sim]["a2"],
        b2=lambda wildcards: sim_config[wildcards.sim]["b2"],
        p=lambda wildcards: sim_config[wildcards.sim]["p"],
        q=lambda wildcards: sim_config[wildcards.sim]["q"],
        w=lambda wildcards: sim_config[wildcards.sim]["w"],
    script:
        "../scripts/simulate_hmm.py"


rule infer_params_hmm:
    """Infer parameters under the HMM model for a supplied composite likelihood."""
    input:
        hmm_data="results/simulations/hmm_joint_gamma/{sim}.rep{seed}.npz",
    output:
        inf_hmm_params="results/simulations/hmm_joint_gamma/{sim}.rep{seed}.params.inferred.tsv",
    params:
        niter=lambda wildcards: sim_config[wildcards.sim]["niter"],
        t=lambda wildcards: sim_config[wildcards.sim]["t"],
    script:
        "../scripts/infer_hmm_params.py"
