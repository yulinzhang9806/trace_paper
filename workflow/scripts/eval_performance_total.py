"""Performance evaluation from an HMM."""

import sys

import numpy as np
import pandas as pd
import tszip
from tqdm import tqdm

sys.path.append(".")
from arg_hmm.utils import Performance_utils  # noqa


def read_truth(bedfile, popmerge=False):
    """Read the truth segments from a bed-file."""
    with open(bedfile) as bed:
        lines = bed.readlines()
        if popmerge:
            out = []
            ind = 0
            while ind < len(lines):
                s = lines[ind].strip("\n").strip("\t").split("\t")
                out.append((float(s[1]), float(s[2])))
                ind += 1
        else:
            out = {}
            ind = 0
            while ind < len(lines):
                s = lines[ind].strip("\n").strip("\t").split("\t")
                if int(s[3]) not in out.keys():
                    out[int(s[3])] = []
                out[int(s[3])].append((float(s[1]), float(s[2])))
                ind += 1
        return out


def generate_treespan(ts):
    """Generate a treespan for a specific tree-sequence of interest.

    Args:
        ts (`tskit.TreeSequence`): tree sequence object for inference

    Returns:
       treespan (`np.array`): two dimensional array

    """
    treespan = []
    for tree in ts.trees():
        treespan.append([tree.interval.left, tree.interval.right])
    return np.array(treespan)


def evaluate_performance(
    pps, truth, treespan, n=10, popmerge=True, min_l=70e3, min_pp=0.90, arc_c=0.5
):
    """Evaluate performance across a number of variety of thresholds.

    Args:
        pps (`np.array`): numpy array containing the posterior probability of
        truth (`list` or `dict`): dictionary containing per-sample or aggregated information
        n (`int`): the number of samples considered

    Returns:
        tot_precision (`float`): total precision of the inference
        tot_recall (`float`): total recall of the inference

    """
    p_utils = Performance_utils()
    st = p_utils.filter_hmm_output(
        combined_pp=pps,
        treespan=treespan,
        arc_cutoff=arc_c,  # NOTE: this is a pre-scan step that filters on posterior
        pp_cutoff=min_pp,  # NOTE: this is the mean posterior threshold for the chunk
        l_cutoff=min_l,
        popmerge=popmerge,
    )
    if popmerge:
        tot_precision, tot_recall, _, _, _ = p_utils.calculate_performance(
            i=st, t=truth
        )
    else:
        cur_precisions = []
        cur_recalls = []
        for g in range(n):
            if g in truth:
                precision, recall, _, _, _ = p_utils.calculate_performance(
                    i=st[g], t=truth[g]
                )
                if len(truth[g]) > 0:
                    cur_precisions.append(precision)
                    cur_recalls.append(recall)
                else:
                    cur_precisions.append(np.nan)
                    cur_recalls.append(np.nan)
        tot_precision = np.nanmean(cur_precisions)
        tot_recall = np.nanmean(cur_recalls)
    return tot_precision, tot_recall


def calculate_hmmix(path, truth, n=2, popmerge=False, min_l=70e3, min_pp=0.90):
    """Evaluate performance of hmmix.

    path: file path to the hmmix output files.

    n: number of individuals included in the calculation. Note here the default is starting from tsk_0.

    """
    p_utils = Performance_utils()
    tp = []
    for i in range(n):
        out = []
        df = pd.read_csv(str(path) + ".tsk_" + str(i) + ".diploid.txt", sep="\s+")
        df1 = df[
            (df["state"] == "Archaic")
            & (df["mean_prob"] >= min_pp)
            & (df["length"] >= min_l)
        ]
        for j in range(len(df1)):
            out.append((df1.iloc[j, 1], df1.iloc[j, 2]))
        if popmerge:
            tp = tp + out
        else:
            tp.append(out)
    if popmerge:
        tot_precision, tot_recall, _, _, _ = p_utils.calculate_performance(
            i=tp, t=truth
        )
    else:
        cur_precisions = []
        cur_recalls = []
        for g in range(n):
            tt = []
            if 2 * g in truth and int(2 * g + 1) in truth:
                tt = truth[2 * g] + truth[2 * g + 1]
            elif 2 * g in truth and not int(2 * g + 1) in truth:
                tt = truth[2 * g]
            elif int(2 * g + 1) in truth and not 2 * g in truth:
                tt = truth[2 * g + 1]
            else:
                tt = None
            if tt is not None:
                precision, recall, _, _, _ = p_utils.calculate_performance(
                    i=tp[g], t=tt
                )
                cur_precisions.append(precision)
                cur_recalls.append(recall)
            else:
                cur_precisions.append(np.nan)
                cur_recalls.append(np.nan)
        tot_precision = np.nanmean(cur_precisions)
        tot_recall = np.nanmean(cur_recalls)
    return tot_precision, tot_recall


def calculate_sprime(file, truth, min_l=70e3, score=0.9 * 111111):
    """Evaluate performance of hmmix.

    file: file path to the sprime output .score file.

    score: filter the output based on score (default: 1e5).

    """
    df = pd.read_csv(str(file), sep="\s+")
    df1 = (
        df[df["SCORE"] >= score][["CHROM", "POS", "SEGMENT", "SCORE"]]
        .groupby(["SEGMENT", "CHROM"])
        .apply(
            lambda x: x.assign(
                length=lambda x: x["POS"].max() - x["POS"].min(),
                end=lambda x: x["POS"].max(),
                start=lambda x: x["POS"].min(),
            )
        )
    )
    df2 = df1[df1["length"] >= min_l]
    df2["tup"] = df2[["start", "end"]].apply(tuple, axis=1)
    tp = list(df2["tup"].unique())
    p_utils = Performance_utils()
    tot_precision, tot_recall, _, _, _ = p_utils.calculate_performance(i=tp, t=truth)
    return tot_precision, tot_recall


if __name__ == "__main__":
    # Define the minimum lengths that we would like to evaluate against
    min_lens = np.linspace(10e3, 200e3, 50)
    # The posterior probability thresholds for evaluation
    min_pp = [0.75, 0.80, 0.90, 0.95, 0.99]
    # Load the posterior results in question.
    data = np.load(snakemake.input["hmm_posterior"])  # noqa
    pps = data["posteriors"]
    pos = data["pos"]
    # Load the empirical filtering use.
    emp_data = np.load(snakemake.input["emp_data"])  # noqa
    upper_bound = emp_data["upper_bound"]
    t_archaic = emp_data["t_archaic"]
    # Load hmmix result
    path_hmmix = snakemake.params["hmmix_pref"]  # noqa
    n_ind = snakemake.params["n_ind_hmmix"]  # noqa
    # Load Sprime result
    path_sprime = snakemake.input["sprime"]  # noqa
    # Load the truth
    true_ts = ts = tszip.decompress(snakemake.input["tsz"])  # noqa
    treespan = generate_treespan(true_ts)
    truth_data_merged = read_truth(snakemake.input["merged_bed"], popmerge=True)  # noqa
    truth_data_indiv = read_truth(snakemake.input["ind_bed"], popmerge=False)  # noqa
    n = snakemake.params["n"]  # noqa

    # Create aggregators for data eventually to be put into data frame
    agg_min_lens = []
    agg_pp = []
    agg_precision_avg = []
    agg_recall_avg = []
    agg_merged_precision = []
    agg_merged_recall = []
    agg_emp_precision_avg = []
    agg_emp_recall_avg = []
    agg_emp_precision_merged = []
    agg_emp_recall_merged = []
    agg_hmmix_precision_avg = []
    agg_hmmix_recall_avg = []
    agg_hmmix_precision_merged = []
    agg_hmmix_recall_merged = []
    agg_sprime_precision_merged = []
    agg_sprime_recall_merged = []
    for m_pp in min_pp:
        for m_l in tqdm(min_lens):
            agg_pp.append(m_pp)
            agg_min_lens.append(m_l)
            cur_precision, cur_recall = calculate_sprime(
                path_sprime, truth_data_merged, min_l=m_l, score=m_pp * 111111
            )
            agg_sprime_precision_merged.append(cur_precision)
            agg_sprime_recall_merged.append(cur_recall)
            for m in [True, False]:
                if m:
                    cur_precision, cur_recall = evaluate_performance(
                        pps,
                        truth_data_merged,
                        treespan,
                        n=n,
                        popmerge=m,
                        min_l=m_l,
                        min_pp=m_pp,
                    )
                    agg_merged_precision.append(cur_precision)
                    agg_merged_recall.append(cur_recall)
                    cur_precision, cur_recall = evaluate_performance(
                        upper_bound,
                        truth_data_merged,
                        treespan,
                        n=n,
                        popmerge=m,
                        min_l=m_l,
                        min_pp=t_archaic,
                        arc_c=t_archaic,
                    )
                    agg_emp_precision_merged.append(cur_precision)
                    agg_emp_recall_merged.append(cur_recall)
                    cur_precision, cur_recall = calculate_hmmix(
                        path_hmmix,
                        truth_data_merged,
                        n=n_ind,
                        popmerge=m,
                        min_l=m_l,
                        min_pp=m_pp,
                    )
                    agg_hmmix_precision_merged.append(cur_precision)
                    agg_hmmix_recall_merged.append(cur_recall)
                else:
                    cur_precision, cur_recall = evaluate_performance(
                        pps,
                        truth_data_indiv,
                        treespan,
                        n=n,
                        popmerge=m,
                        min_l=m_l,
                        min_pp=m_pp,
                    )
                    agg_precision_avg.append(cur_precision)
                    agg_recall_avg.append(cur_recall)
                    cur_precision, cur_recall = evaluate_performance(
                        upper_bound,
                        truth_data_indiv,
                        treespan,
                        n=n,
                        popmerge=m,
                        min_l=m_l,
                        min_pp=t_archaic,
                        arc_c=t_archaic,
                    )
                    agg_emp_precision_avg.append(cur_precision)
                    agg_emp_recall_avg.append(cur_recall)
                    cur_precision, cur_recall = calculate_hmmix(
                        path_hmmix,
                        truth_data_indiv,
                        n=n_ind,
                        popmerge=m,
                        min_l=m_l,
                        min_pp=m_pp,
                    )
                    agg_hmmix_precision_avg.append(cur_precision)
                    agg_hmmix_recall_avg.append(cur_recall)

    # Generate the data frame and write out to a file
    perf_df = pd.DataFrame(
        {
            "min_length": agg_min_lens,
            "min_posterior_prob": agg_pp,
            "precision_avg": agg_precision_avg,
            "recall_avg": agg_recall_avg,
            "precision_merged": agg_merged_precision,
            "recall_merged": agg_merged_recall,
            "precision_emp_avg": agg_emp_precision_avg,
            "recall_emp_avg": agg_emp_recall_avg,
            "precision_emp_merged": agg_emp_precision_merged,
            "recall_emp_merged": agg_emp_recall_merged,
            "precision_hmmix_avg": agg_hmmix_precision_avg,
            "recall_hmmix_avg": agg_hmmix_recall_avg,
            "precision_hmmix_merged": agg_hmmix_precision_merged,
            "recall_hmmix_merged": agg_hmmix_recall_merged,
            "precision_sprime": agg_sprime_precision_merged,
            "recall_sprime": agg_sprime_recall_merged,
        }
    )
    perf_df.to_csv(snakemake.output["performance_stats"], sep="\t", index=False)  # noqa
