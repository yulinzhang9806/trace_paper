"""Utils for running simulation-based analysis workflow."""
import json

import numpy as np
import pandas as pd
import tskit
import tszip
from tqdm import tqdm
from pathlib import Path
import sys
from joblib import Parallel, delayed

from arg_hmm.arg_hmm import GhostProductHmm
from arg_hmm.utils import ARG_utils, Performance_utils
import os
import pybedtools
from pyfaidx import Fasta


class Analysis_workflow_utils:
    """A class of functions used in analysis workflow."""
    
    def run_hmm(self, ncoal, treespan, intro_prop, seed):
        """A helper function to run GhostProductHMM."""
        hmm = GhostProductHmm()
        hmm.init_hmm(ncoal, treespan, intro_prop=intro_prop)
        res_dict = hmm.train(seed=seed)
        gammas, alphas, betas = hmm.decode()
        return np.exp(gammas), res_dict, hmm.treespan

    def perf(self, idx, tru_seg, inferred, fdc=False):
        """A helper function to run performance evaluation."""
        performance = Performance_utils()
        if fdc:
            lf = sum([x[1] - x[0] for x in inferred[idx]])
            pre = lf / tru_seg
            rec = lf / tru_seg
            tru_seq = 0
            total_HMM = lf
        else:
            arg_utils = ARG_utils()
            truth = arg_utils.get_ghost_intro_ind(i=idx, all_intro=tru_seg)
            truth = [tuple(truth[i]) for i in range(len(truth))]
            pre, rec, _, _, tru_seq, total_HMM = performance.calculate_performance(
                inferred[idx], truth
            )
        return pre, rec, tru_seq, total_HMM

    def record_hmm_results_ondind(
        self,
        ind,
        target_tsz,
        outgroup_tsz,
        pp_output,
        t_archaic,
        intro_prop,
        seed,
    ):
        """Run GhostHMM on target and outgroup populations and record results.

        sample_inds: if None, use all individuals in the tree to run ECM. Otherwise use sample_inds + target_ind
        to run ECM and decode only target_ind.
        """
        target_ts = ""
        outgroup_ts = ""
        target_sub_ts = False
        if target_tsz.endswith('.tsz'):
            target_ts = tszip.decompress(str(target_tsz))
            outgroup_ts = tszip.decompress(str(outgroup_tsz))
        else:
            target_ts = tskit.load(str(target_tsz))
            outgroup_ts = tskit.load(str(outgroup_tsz))
        
        n_gamma = np.zeros(shape=(2, target_ts.num_trees))
        f_gamma = np.zeros(shape=(2, outgroup_ts.num_trees))

        # init hmm for target population
        ahmm = GhostProductHmm()
        ancoal, at1s, at2s, atreespan, anleaves = ahmm.prepare_data_tmrca(ts = target_ts, ind = ind, t_archaic=t_archaic)
        n_gamma, aoutdict, atreespan = self.run_hmm(ncoal = ancoal, treespan = atreespan, intro_prop = intro_prop, seed = seed)
        a_param = pd.DataFrame.from_dict(aoutdict).to_numpy()

        # init hmm for outgroup population
        fhmm = GhostProductHmm()
        fncoal, ft1s, ft2s, ftreespan, fnleaves = fhmm.prepare_data_tmrca(ts = outgroup_ts, ind = ind, t_archaic=t_archaic)
        f_gamma, foutdict, ftreespan = self.run_hmm(ncoal = fncoal, treespan = ftreespan, intro_prop = intro_prop, seed = seed)
        f_param = pd.DataFrame.from_dict(foutdict).to_numpy()
        
        np.savez_compressed(
            str(pp_output) + "ind" + str(ind) + ".npz",
            ancoal=ancoal,
            at1s=at1s,
            at2s=at2s,
            anleaves=anleaves,
            fncoal=fncoal,
            ft1s=ft1s,
            ft2s=ft2s,
            fnleaves=fnleaves,
            n_gamma=n_gamma,
            f_gamma=f_gamma,
            a_param=a_param,
            f_param=f_param,
            a_tree_span=atreespan,
            f_tree_span=ftreespan,
            allow_pickle=True
        )

    def record_performance(
        self,
        inds,
        ind_bed,
        n,
        pp_outpref,
        l_cutoff,
        pp_cutoff,
        sequence_length,
        intro_prop,
        perf_output,
        maxlen=None,
    ):
        """Record performance of GhostHMM on target and outgroup populations."""
        # read in pp outputs for each ind
        data = np.load(pp_outpref + "0.npz", allow_pickle=True)
        tree_span = data["a_tree_span"]
        target_pp = np.zeros(shape=(inds, data["n_gamma"].shape[1]))
        tree_span_f = data["f_tree_span"]
        outgroup_pp = np.zeros(shape=(inds, data["f_gamma"].shape[1]))
        for i in range(inds):
            data = np.load(pp_outpref + str(i) + ".npz", allow_pickle=True)
            target_pp[i] = data["n_gamma"][1, :]
            outgroup_pp[i] = data["f_gamma"][1, :]

        out_perf = {
            "individual": [],
            "mean_posterior_cutoff": [],
            "length_cutoff": [],
            "precision": [],
            "recall": [],
            "false_discovery": [],
            "total_detection": [],
            "length_true": [],
            "fdc_length": [],
        }
        all_truth = Performance_utils().read_truth_bed(ind_bed, 2 * int(n))
        for l in range(len(l_cutoff)):
            for pff in range(len(pp_cutoff)):
                inferred_target, _ = Performance_utils().filter_hmm_output(
                    arc_cutoff=0.5,
                    pp_cutoff=pp_cutoff[pff],
                    l_cutoff=l_cutoff[l],
                    combined_pp=target_pp,
                    treespan=tree_span,
                    maxlen=maxlen,
                )
                inferred_out, _ = Performance_utils().filter_hmm_output(
                    arc_cutoff=0.5,
                    pp_cutoff=pp_cutoff[pff],
                    l_cutoff=l_cutoff[l],
                    combined_pp=outgroup_pp,
                    treespan=tree_span_f,
                    maxlen=maxlen,
                )
                for idx in range(inds):
                    pre, rec, tru_seq, total_HMM = self.perf(
                        idx=idx,
                        tru_seg=all_truth,
                        inferred=inferred_target,
                        fdc=False,
                    )
                    fdc, _, _, lfdc = self.perf(
                        idx=idx,
                        tru_seg=float(intro_prop) * sequence_length * 1e-6,
                        inferred=inferred_out,
                        fdc=True,
                    )
                    out_perf["individual"].append(idx)
                    out_perf["mean_posterior_cutoff"].append(pp_cutoff[pff])
                    out_perf["length_cutoff"].append(l_cutoff[l])
                    out_perf["precision"].append(pre)
                    out_perf["recall"].append(rec)
                    out_perf["false_discovery"].append(fdc)
                    out_perf["total_detection"].append(total_HMM)
                    out_perf["length_true"].append(tru_seq)
                    out_perf["fdc_length"].append(lfdc)
        out_pref = pd.DataFrame.from_dict(out_perf)
        out_pref.to_csv(perf_output, sep="\t", index=False)

    def ibdmix_performance(
        self,
        inds,
        inputf,
        inputf_null,
        exp_intro,
        ind_bed,
        n,
        l_cutoff,
        pp_cutoff,
        slod_cutoff,
        perf_output,
    ):
        """Evaluate performance of ibdmix."""
        out_perf = {
            "individual": [],
            "mean_posterior_cutoff": [],
            "length_cutoff": [],
            "precision": [],
            "recall": [],
            "false_discovery": [],
            "total_detection": [],
            "length_true": [],
            "fdc_length": [],
        }
        df = pd.read_csv(
            str(inputf), sep="\s+"
        )
        df["length"] = (df['end'] - df['start']) / 1e6
        df1 = pd.read_csv(
            str(inputf_null), sep="\s+"
        )
        df1["length"] = (df1['end'] - df1['start']) / 1e6
        for ppindex, pp in enumerate(slod_cutoff):
            for l in l_cutoff:
                tp = []
                fdc = []
                for i in range(int(inds)):
                    out = []
                    dff = df[(df['ID'] == f"tsk_{i}") & (df["length"] > l)]
                    dff = dff[dff["slod"] >= pp]
                    for j in range(len(dff)):
                        out.append((dff.iloc[j, 2] / 1e6, dff.iloc[j, 3] / 1e6))
                    tp.append(out)
                    
                    out = []
                    dff = df1[(df1['ID'] == f"tsk_{i}") & (df1["length"] > l)]
                    dff = dff[dff["slod"] >= pp]
                    for j in range(len(dff)):
                        out.append((dff.iloc[j, 2] / 1e6, dff.iloc[j, 3] / 1e6))
                    fdc.append(out)
                cur_precisions = []
                cur_recalls = []
                cur_tru = []
                cur_hmm = []
                cur_fdc = []
                truth = Performance_utils().read_truth_bed(ind_bed, 2 * int(n))
                for g in range(int(inds)):
                    tt = []
                    if len(truth[2 * g]) > 0 and len(truth[2 * g + 1]) > 0:
                        tt = np.append(truth[2 * g], truth[2 * g + 1], axis=0)
                    if len(truth[2 * g]) == 0:
                        tt = truth[2 * g + 1]
                    if len(truth[2 * g + 1]) == 0:
                        tt = truth[2 * g]
                    if len(tt) > 0:
                        (
                            precision,
                            recall,
                            _,
                            _,
                            tru_seq,
                            total_HMM,
                        ) = Performance_utils().calculate_performance(i=tp[g], t=tt)
                        cur_precisions.append(precision)
                        cur_recalls.append(recall)
                        cur_tru.append(tru_seq)
                        cur_hmm.append(total_HMM)
                    else:
                        cur_precisions.append(np.nan)
                        cur_recalls.append(np.nan)
                        cur_tru.append(np.nan)
                        cur_hmm.append(np.nan)
                    if len(fdc[g]) > 0:
                        fdc_length = sum([x[1] - x[0] for x in fdc[g]])
                        cur_fdc.append(fdc_length)
                    else:
                        cur_fdc.append(np.nan)
                out_perf["individual"] = out_perf["individual"] + list(range(int(inds)))
                out_perf["mean_posterior_cutoff"] = out_perf[
                    "mean_posterior_cutoff"
                ] + [pp_cutoff[ppindex]] * int(inds)
                out_perf["length_cutoff"] = out_perf["length_cutoff"] + [l] * int(inds)
                out_perf["precision"] = out_perf["precision"] + cur_precisions
                out_perf["recall"] = out_perf["recall"] + cur_recalls
                out_perf["false_discovery"] = out_perf["false_discovery"] + (np.array(cur_fdc) / exp_intro).tolist()
                out_perf["total_detection"] = out_perf["total_detection"] + cur_hmm 
                out_perf["length_true"] = out_perf["length_true"] + cur_tru
                out_perf["fdc_length"] = out_perf["fdc_length"] + cur_fdc
        out_pref = pd.DataFrame.from_dict(out_perf)
        out_pref.to_csv(perf_output, sep="\t", index=False)

    def hmmix_performance(
        self,
        inds,
        inputpref,
        inputpref_null,
        exp_intro,
        ind_bed,
        n,
        l_cutoff,
        pp_cutoff,
        perf_output,
    ):
        """Evaluate performance of hmmix."""
        out_perf = {
            "individual": [],
            "mean_posterior_cutoff": [],
            "length_cutoff": [],
            "precision": [],
            "recall": [],
            "false_discovery": [],
            "total_detection": [],
            "length_true": [],
            "fdc_length": [],
        }
        for pp in pp_cutoff:
            for l in l_cutoff:
                tp = []
                fdc = []
                for i in range(int(inds)):
                    out = []
                    df = pd.read_csv(
                        str(inputpref) + str(int(i / 2)) + ".hap" + str(int(i%2) + 1) + ".txt", sep="\s+"
                    )
                    df["length"] = df["length"] / 1e6
                    df1 = df[
                        (df["state"] == "Archaic")
                        & (df["mean_prob"] >= pp)
                        & (df["length"] >= l)
                    ]
                    for j in range(len(df1)):
                        out.append((df1.iloc[j, 1] / 1e6, (df1.iloc[j, 2] + 1000) / 1e6))
                    tp.append(out)
                    out = []
                    df1 = pd.read_csv(
                       str(inputpref_null) + str(int(i / 2)) + ".hap" + str(int(i%2) + 1) + ".txt", sep="\s+"
                    )
                    df1["length"] = df1["length"] / 1e6
                    df1 = df1[
                        (df1["state"] == "Archaic")
                        & (df1["mean_prob"] >= pp)
                        & (df1["length"] >= l)
                    ]
                    for j in range(len(df1)):
                        out.append((df1.iloc[j, 1] / 1e6, (df1.iloc[j, 2] + 1000) / 1e6))
                    fdc.append(out)
                cur_precisions = []
                cur_recalls = []
                cur_tru = []
                cur_hmm = []
                cur_fdc = []
                truth = Performance_utils().read_truth_bed(ind_bed, 2 * int(n))
                for g in range(int(inds)):
                    tt = truth[g]
                    if len(tt) > 0:
                        (
                            precision,
                            recall,
                            _,
                            _,
                            tru_seq,
                            total_HMM,
                        ) = Performance_utils().calculate_performance(i=tp[g], t=tt)
                        cur_precisions.append(precision)
                        cur_recalls.append(recall)
                        cur_tru.append(tru_seq)
                        cur_hmm.append(total_HMM)
                    else:
                        cur_precisions.append(np.nan)
                        cur_recalls.append(np.nan)
                        cur_tru.append(np.nan)
                        cur_hmm.append(np.nan)
                    if len(fdc[g]) > 0:
                        fdc_length = sum([x[1] - x[0] for x in fdc[g]])
                        cur_fdc.append(fdc_length)
                    else:
                        cur_fdc.append(np.nan)
                out_perf["individual"] = out_perf["individual"] + list(range(int(inds)))
                out_perf["mean_posterior_cutoff"] = out_perf[
                    "mean_posterior_cutoff"
                ] + [pp] * int(inds)
                out_perf["length_cutoff"] = out_perf["length_cutoff"] + [l] * int(inds)
                out_perf["precision"] = out_perf["precision"] + cur_precisions
                out_perf["recall"] = out_perf["recall"] + cur_recalls
                out_perf["false_discovery"] = out_perf["false_discovery"] + (np.array(cur_fdc) / exp_intro).tolist()
                out_perf["total_detection"] = out_perf["total_detection"] + cur_hmm 
                out_perf["length_true"] = out_perf["length_true"] + cur_tru
                out_perf["fdc_length"] = out_perf["fdc_length"] + cur_fdc
        out_pref = pd.DataFrame.from_dict(out_perf)
        out_pref.to_csv(perf_output, sep="\t", index=False)

    def calculate_sprime(self, file, fdr_file, truth, min_l=70e3, score=0.9 * 1e5):
        """Evaluate performance of sprime.

        file: file path to the sprime output .score file.

        score: filter the output based on score (default: 1e5).

        """
        df = pd.read_csv(str(file), sep="\s+")
        df["POS"] = df["POS"] / 1e6
        dfdr = pd.read_csv(str(fdr_file), sep="\s+")
        dfdr["POS"] = dfdr["POS"] / 1e6
        if len(df[df["SCORE"] >= float(score)]) > 0:
            df1 = (
                df[df["SCORE"] >= float(score)][["CHROM", "POS", "SEGMENT", "SCORE"]]
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
        else:
            tp = []
        if len(dfdr[dfdr["SCORE"] >= float(score)]) > 0:
            df1 = (
                dfdr[dfdr["SCORE"] >= float(score)][["CHROM", "POS", "SEGMENT", "SCORE"]]
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
            fdc = list(df2["tup"].unique())
        else:
            fdc = []
        p_utils = Performance_utils()
        tot_precision, tot_recall, _, _, _, _ = p_utils.calculate_performance(
            i=tp, t=truth
        )
        tot_fdc = sum([x[1] - x[0] for x in fdc])
        return tot_precision, tot_recall, tot_fdc

    def record_hmm_data_singer_singlets(
        self,
        ind,
        targetpref,
        outgrouppref,
        asamp,
        windowsize,
        t_archaic,
        pp_output,
        func = np.mean,
    ):
        """Extract data for hmm training."""
        def get_data(ts, ind, t_archaic, windowsize, func):
            genome_length = ts.sequence_length
            m = int(genome_length / windowsize) + int(genome_length % windowsize > 0)
            ncoal_sub = np.zeros((len(ind), m))
            t1s_sub = np.zeros((len(ind), m))
            t2s_sub = np.zeros((len(ind), m))
            nleaves_sub = np.zeros((len(ind), m))
            hmm = GhostProductHmm()
            tncoal, tt1s, tt2s, treespan, tnleaves = hmm.prepare_data_tmrca(ts = ts, ind = ind, t_archaic=t_archaic)
            if len(ind) == 1:
                tncoal = np.array([tncoal])
                tt1s = np.array([tt1s])
                tt2s = np.array([tt2s])
                tnleaves = np.array([tnleaves])
            for i in range(len(ind)):
                t = 0
                curtrees = []
                for k in range(m):
                    while t < treespan.shape[0] and treespan[t][0] < k * windowsize + windowsize:
                        curtrees.append(t)
                        t += 1
                    if len(curtrees) == 0:
                        ncoal_sub[i][k] = tncoal[i][t - 1]
                        t1s_sub[i][k] = tt1s[i][t - 1]
                        t2s_sub[i][k] = tt2s[i][t - 1]
                        nleaves_sub[i][k] = tnleaves[i][t - 1]
                    else:
                        treelens = []
                        curtrees = np.array(curtrees)
                        for j in range(len(curtrees)):
                            treelens.append(min(treespan[curtrees[j]][1], int(k * windowsize + windowsize)) - max(treespan[curtrees[j]][0], int(k * windowsize)))
                        treelens = np.array(treelens)
                        curtrees = curtrees[treelens > 1]
                        treelens = treelens[treelens > 1]
                        if len(curtrees) == 0:
                            ncoal_sub[i][k] = tncoal[i][t - 1]
                            t1s_sub[i][k] = tt1s[i][t - 1]
                            t2s_sub[i][k] = tt2s[i][t - 1]
                            nleaves_sub[i][k] = tnleaves[i][t - 1]
                        else:
                            treelens = (treelens / np.min(treelens)).astype(int)
                            ncoal_sub[i][k] = np.average(tncoal[i][curtrees], weights=treelens)
                            t1s_sub[i][k] = np.average(tt1s[i][curtrees], weights=treelens)
                            t2s_sub[i][k] = np.average(tt2s[i][curtrees], weights=treelens)
                            nleaves_sub[i][k] = np.average(tnleaves[i][curtrees], weights=treelens)
                        curtrees = []
                        if treespan[t-1][1] < (k+1) * windowsize + windowsize:
                            curtrees.append(t - 1)
            return ncoal_sub, t1s_sub, t2s_sub, nleaves_sub
        if isinstance(ind, int):
            ind = [ind]
        target_ts = tszip.decompress(str(targetpref) + str(asamp) + ".tsz")
        ancoal_sub, at1s_sub, at2s_sub, anleaves_sub = get_data(target_ts, ind, t_archaic, windowsize, func)
        atreespan = np.array([[t*windowsize, (t + 1) * windowsize] for t in range(ancoal_sub.shape[1])])
        outgroup_ts = tszip.decompress(str(outgrouppref) + str(asamp) + ".tsz")
        fncoal_sub, ft1s_sub, ft2s_sub, fnleaves_sub = get_data(outgroup_ts, ind, t_archaic, windowsize, func)
        ftreespan = np.array([[t*windowsize, (t + 1) * windowsize] for t in range(fncoal_sub.shape[1])])
        np.savez_compressed(
            str(pp_output) + str(asamp) + ".npz",
            ancoal=ancoal_sub,
            at1s=at1s_sub,
            at2s=at2s_sub,
            anleaves=anleaves_sub,
            fncoal=fncoal_sub,
            ft1s=ft1s_sub,
            ft2s=ft2s_sub,
            fnleaves=fnleaves_sub,
            a_tree_span=atreespan,
            f_tree_span=ftreespan,
            allow_pickle=True
        )

    def record_hmm_results_avg_oneind(
        self,
        ind,
        npz_filepath,
        fsample,
        pp_output,
        t_archaic,
        intro_prop,
        func = np.mean,
        seed = 1,
    ):
        """Run GhostHMM on target and outgroup populations and record results by .npz file.
        sample_inds: if None, use all individuals in the tree to run ECM. Otherwise use sample_inds + target_ind
        to run ECM and decode only target_ind.
        """
        # init hmm for target population
        for idx, f in enumerate(fsample):
            data = np.load(str(npz_filepath) + str(f) + ".npz", allow_pickle=True)
            if idx == 0:
                ancoal = np.zeros(shape=(len(fsample), data["ancoal"].shape[1]))
                fncoal = np.zeros(shape=(len(fsample), data["fncoal"].shape[1]))
                at1s = np.zeros(shape=(len(fsample), data["at1s"].shape[1]))
                ft1s = np.zeros(shape=(len(fsample), data["ft1s"].shape[1]))
                at2s = np.zeros(shape=(len(fsample), data["at2s"].shape[1]))
                ft2s = np.zeros(shape=(len(fsample), data["ft2s"].shape[1]))
                anleaves = np.zeros(shape=(len(fsample), data["anleaves"].shape[1]))
                fnleaves = np.zeros(shape=(len(fsample), data["fnleaves"].shape[1]))
            ancoal[idx] = data["ancoal"][ind]
            at1s[idx] = data["at1s"][ind]
            at2s[idx] = data["at2s"][ind]
            anleaves[idx] = data["anleaves"][ind]
            fncoal[idx] = data["fncoal"][ind]
            ft1s[idx] = data["ft1s"][ind]
            ft2s[idx] = data["ft2s"][ind]
            fnleaves[idx] = data["fnleaves"][ind]
        atreespan = data["a_tree_span"]
        ftreespan = data["f_tree_span"]
        n_gamma, outdict, atreespan = self.run_hmm(
            ncoal = func(ancoal, axis = 0),
            treespan = atreespan,
            intro_prop = intro_prop,
            seed = seed,
        )
        a_param = pd.DataFrame.from_dict(outdict).to_numpy()
        f_gamma, foutdict, ftreespan = self.run_hmm(
            ncoal = func(fncoal, axis = 0),
            treespan = ftreespan,
            intro_prop = intro_prop,
            seed = seed,
        )
        f_param = pd.DataFrame.from_dict(foutdict).to_numpy()
        np.savez_compressed(
            str(pp_output) + "ind" + str(ind) + ".npz",
            ancoal=ancoal,
            at1s=at1s,
            at2s=at2s,
            anleaves=anleaves,
            fncoal=fncoal,
            ft1s=ft1s,
            ft2s=ft2s,
            fnleaves=fnleaves,
            n_gamma=n_gamma,
            f_gamma=f_gamma,
            a_param=a_param,
            f_param=f_param,
            a_tree_span=atreespan,
            f_tree_span=ftreespan,
            allow_pickle=True
        )


class SNPINFO:
    """A class of functions to get SNP ancestral and archaic information."""
    
    ## checked
    def get_snp_info(self, bcffile, ancesterfa, outpref):
        """Get SNP info file with polarization information.
    
        bcffile: a file end with ".bcf" and could be read by bcftools.
        ancesterfa: a fasta file of ancestral genome, end with ".fa"

        return: a txt file with snp information
        """
        if os.path.exists(str(outpref) + ".txt"):
            os.remove(str(outpref) + ".txt")
        # ignore indels
        os.system(
            "bcftools view -v snps " + str(bcffile) + " | bcftools query -f '%CHROM\t%POS\t%REF\t%ALT\n' > " + str(outpref) + ".bcftools_temp.txt"
        )
        type_dict = {'AT':'TA', 'TA':'TA', 'TC':'TC', 'AG':'TC', 'TG':'TG', 'AC':'TG', 'CT':'CT', 'GA':'CT', 'CA':'CA', 'GT':'CA', 'CG':'CG', 'GC':'CG', 'NN':'NN'}
        ancestral = Fasta(ancesterfa)
        c = list(ancestral.keys())[0]
        infile=open(str(outpref) + ".bcftools_temp.txt")
        lines=infile.readlines()
        infile.close()
        out = "chr\tpos\tref\talt\tancestral\tderived\tupstream\tdownstream\ttype\ttype_fold\tCpG\n"
        for i in range(len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            pos = int(s[1])
            ref = str(s[2])
            if len(ref) > 1: # skip indels
                continue
            alts = str(s[3]).strip('\t').strip(',').split(',')
            for alt in alts:
                out += "\t".join(lines[i].strip('\n').split('\t')[:-1]) + '\t' + alt + '\t'
                triplit = ancestral[c][pos-2:pos+1].seq
                up = list(triplit)[0]
                anc = list(triplit)[1]
                down = list(triplit)[2]
                out += anc.upper() + '\t'
                dr = 'N'
                if anc.upper() == ref.upper():
                    dr = alt.upper()
                    out += dr + '\t'
                elif anc.upper() == alt.upper():
                    dr = ref.upper()
                    out += dr + '\t'
                elif anc.upper() in ['A', 'T', 'C', 'G']:
                    dr = ref.upper() + ',' + alt.upper()
                    out += dr + '\t'
                else:
                    anc = 'N'
                    out += 'N\t'
                out += up.upper() + '\t' + down.upper() + '\t'
                if len(dr) == 1:
                    ty = anc.upper() + dr.upper()
                else:
                    ty = anc.upper() + dr.split(',')[0] + ',' + anc.upper() + dr.split(',')[1]
                out += ty + '\t'
                try:
                    out += type_dict[ty] + '\t'
                except:
                    out += type_dict[ty.split(',')[0]] + ',' + type_dict[ty.split(',')[1]] + '\t'
                if anc.upper() == 'C' and down.upper() == 'G':
                    out += str(1)
                elif anc.upper() == 'G' and up.upper() == 'C':
                    out += str(1)
                else:
                    out += str(0)
                out += '\n'
        outfile=open(outpref + '.txt','w')
        outfile.write(out)
        outfile.close()
        os.remove(str(outpref) + ".bcftools_temp.txt")

    ## checked
    def parse_archaic_geno(self, archaic_geno, arc_return_dict):
        """Helper function to parse archaic_geno file.

        archaic_geno: a file of "bcftools query -f'[%CHROM\t%POS\t%REF\t%ALT\t%SAMPLE\t%GT\n]'" output
        arc_return_dict: a dictionary indicating the output genotype order

        return: a set and a dictionary (pos:[genotypes(order by arc_return_dict), ref, alt])
        """
        geno_dict = {'0/0':0, '0/1':1, '1/0':1, '1/1':2, './.':9, './0':9, '0/.':9, './1':9, '1/.':9}
        infile=open(archaic_geno)
        lines=infile.readlines()
        infile.close()
        existing_sites = set()
        geno_info = dict()
        for i in range(len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            if len(str(s[3])) > 1: # archaics are multiallelic at the site, ignore
                continue
            if not "_".join([s[1], s[2], s[3]]) in existing_sites:
                existing_sites.add("_".join([s[1], s[2], s[3]]))
                geno_info["_".join([s[1], s[2], s[3]])] = [0, 0, 0, 0, 0, 0]
                geno_info["_".join([s[1], s[2], s[3]])][4] = str(s[2])
                geno_info["_".join([s[1], s[2], s[3]])][5] = str(s[3])
            geno_info["_".join([s[1], s[2], s[3]])][arc_return_dict[str(s[4])]] = geno_dict[str(s[5])]
        return existing_sites, geno_info

    ## checked
    def append_archaic_snpinfo(self, snpinfo, archaicbcf, outpref):
        """Append archaic snp info to the existing snpinfo file.
        
        snpinfo: str, no file extension
        archaicbcf: str, no file extension

        return: new snpinfo txt file.
        """
            
        os.system(
            "cut -f 1,2 " + str(snpinfo) + ".txt | tail -n +2 > " + snpinfo + "allsnp"
        )
        # used split multiallelic site versions here
        os.system(
            "bcftools query -f'[%CHROM\t%POS\t%REF\t%ALT\t%SAMPLE\t%GT\n]' -R " + snpinfo + "allsnp " + str(archaicbcf) + ".bcf" + " > " + snpinfo + "archaic_geno"
        )
        os.remove(snpinfo + "allsnp")
        arc_return_dict = {'Chagyrskaya-Phalanx':0, 'AltaiNeandertal':1, 'Vindija33.19':2, 'Denisova':3}
        existing_sites, geno_info = self.parse_archaic_geno(snpinfo + "archaic_geno", arc_return_dict)
        infile=open(snpinfo + '.txt')
        lines=infile.readlines()
        infile.close()
        out = lines[0].strip('\n') + '\tChagyrskaya-Phalanx\tAltaiNeandertal\tVindija33.19\tDenisova\n'
        for i in range(1, len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            out += lines[i].strip('\n') + '\t'
            if "_".join([s[1], s[2], s[3]]) in existing_sites:
                ginfo = geno_info["_".join([s[1], s[2], s[3]])]
            elif "_".join([s[1], s[2], "."]) in existing_sites:
                ginfo = geno_info["_".join([s[1], s[2], "."])]
            else:
                out += "9\t9\t9\t9\n"
                continue
            if ginfo[4].upper() == str(s[2]) and (ginfo[5].upper() == str(s[3]) or ginfo[5] == '.'):
                if len(str(s[5])) > 1:
                    out += "2\t2\t2\t2\n"
                else:
                    out += str(ginfo[arc_return_dict['Chagyrskaya-Phalanx']]) + '\t'
                    out += str(ginfo[arc_return_dict['AltaiNeandertal']]) + '\t'
                    out += str(ginfo[arc_return_dict['Vindija33.19']]) + '\t'
                    out += str(ginfo[arc_return_dict['Denisova']]) + '\n'
            else:
                out += "REF_DONT_MATCH\tREF_DONT_MATCH\tREF_DONT_MATCH\tREF_DONT_MATCH\n"
        outfile=open(outpref + '.txt','w')
        outfile.write(out)
        outfile.close()
        os.remove(snpinfo + "archaic_geno")
        return existing_sites, geno_info
    
    ## checked
    def parse_AFR_info(self, bcffile, tempfile, poplabelfile):
        """Helper function to parse AFR SNP info."""
        os.system(
            "bcftools view -v snps -Ov " + str(bcffile) + "| vcftools --vcf - --keep "+str(poplabelfile)+" --freq --stdout > " + str(tempfile)
        )
        infile = open(str(tempfile))
        lines = infile.readlines()
        infile.close()
        afr_dict = dict()
        afr_set = set()
        for i in range(1, len(lines)):
            s = lines[i].strip('\n').strip('\t').split('\t')
            nalleles = int(s[2])
            afr_set.add(int(s[1]))
            if not int(s[1]) in afr_dict:
                afr_dict[int(s[1])] = dict()
            for j in range(len(s) - nalleles, len(s)):
                allele, freq = s[j].split(':')
                afr_dict[int(s[1])][allele] = float(freq)
        os.remove(str(tempfile))
        return afr_dict, afr_set

    ## checked
    def append_AFR_info(self, snpinfo, bcffile, poplabelfile, linelabel, outpref):
        """Append AFR info to the existing snpinfo file."""
        afr_dict, afr_set = self.parse_AFR_info(bcffile, outpref + "afr_frq.txt", poplabelfile)
        infile=open(snpinfo + '.txt')
        lines=infile.readlines()
        infile.close()
        out = lines[0].strip('\n') + '\t' + str(linelabel) + '\n'
        for i in range(1, len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            out += lines[i].strip('\n') + '\t'
            if int(s[1]) in afr_set:
                out += str(afr_dict[int(s[1])][str(s[3])]) + '\n'
            else:
                out += "missing_site\n"
        outfile=open(outpref + '.txt','w')
        outfile.write(out)
        outfile.close()

    def parse_outgroup_hmmix(self, outgroupfile):
        """Parse outgroup info from hmmix outgroup file."""
        infile=open(outgroupfile)
        lines=infile.readlines()
        infile.close()
        outgroupset = set()
        for i in range(1, len(lines)):
            s = lines[i].strip('\n').strip('\t').split('\t')
            outgroupset.add("_".join([str(s[0]), str(s[1])]))
        return outgroupset

    def append_outgroup_info(self, snpinfo, outgroupfile, outpref):
        """Append outgroup info to the existing snpinfo file."""
        outgroupset = self.parse_outgroup_hmmix(outgroupfile)
        infile=open(snpinfo + '.txt')
        lines=infile.readlines()
        infile.close()
        out = lines[0].strip('\n') + '\tin_outgroup\n'
        for i in range(1, len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            out += lines[i].strip('\n') + '\t'
            if "_".join([s[0], s[1]]) in outgroupset:
                out += '1\n'
            else:
                out += '0\n'
        outfile=open(outpref + '.txt','w')
        outfile.write(out)
        outfile.close()

    def parse_strictmask(self, strictmask, bcffile, tempfile):
        """Parse mutations in strictmask."""
        os.system(
            "bcftools view -v snps -R " + str(strictmask) + " " + str(bcffile) + " | bcftools query -f '%CHROM\t%POS\n' > " + str(tempfile)
        )
        infile=open(tempfile)
        lines=infile.readlines()
        infile.close()
        strictmaskset = set()
        for i in range(len(lines)):
            s = lines[i].strip('\n').strip('\t').split('\t')
            strictmaskset.add("_".join([str(s[0]), str(s[1])]))
        os.remove(tempfile)
        return strictmaskset

    def append_strictmask_info(self, snpinfo, strictmask, bcffile, outpref):
        """Append strictmask info to the existing snpinfo file."""
        strictmaskset = self.parse_strictmask(strictmask, bcffile, outpref + "strictmask.txt")
        infile=open(snpinfo + '.txt')
        lines=infile.readlines()
        infile.close()
        out = lines[0].strip('\n') + '\tin_strictmask\n'
        for i in range(1, len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            out += lines[i].strip('\n') + '\t'
            if "_".join([s[0], s[1]]) in strictmaskset:
                out += '1\n'
            else:
                out += '0\n'
        outfile=open(outpref + '.txt','w')
        outfile.write(out)
        outfile.close()

    def append_manifesto_info(self, snpinfo, manifesto, bcffile, outpref):
        """Append strictmask info to the existing snpinfo file."""
        strictmaskset = self.parse_strictmask(manifesto, bcffile, outpref + "manifesto.txt")
        infile=open(snpinfo + '.txt')
        lines=infile.readlines()
        infile.close()
        out = lines[0].strip('\n') + '\tin_manifesto\n'
        for i in range(1, len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            out += lines[i].strip('\n') + '\t'
            if "_".join([s[0], s[1]]) in strictmaskset:
                out += '1\n'
            else:
                out += '0\n'
        outfile=open(outpref + '.txt','w')
        outfile.write(out)
        outfile.close()

    def append_mutage_info(self, snpinfo, mutage_pref, mutage_range, outpref):
        """Append mutage info to the existing snpinfo file."""
        mutage = {}
        for i in mutage_range:
            mutage_sub = {}
            infile = open(str(mutage_pref) + str(i) + ".txt")
            lines = infile.readlines()
            infile.close()
            for j in range(1, len(lines)):
                s = lines[j].strip('\n').strip('\t').split('\t')
                ss = s[2].split('_')
                l = float(ss[0])
                h = float(ss[1])
                if int(s[1]) not in mutage_sub: # get uniquely mapped mutations
                    if l == h:
                        mutage_sub[int(s[1])] = np.nan # remove map to root node mutations
                    else:
                        mutage_sub[int(s[1])] = (l + h) / 2
                else:
                    mutage_sub[int(s[1])] = np.nan
            for k, v in mutage_sub.items():
                if not k in mutage:
                    mutage[k] = []
                mutage[k].append(v)
        infile=open(snpinfo + '.txt')
        lines=infile.readlines()
        infile.close()
        out = lines[0].strip('\n') + '\tmutage\n'
        for i in range(1, len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            out += lines[i].strip('\n') + '\t'
            if int(s[1]) in mutage:
                if len(mutage[int(s[1])]) > 0:
                    out += str(np.nanmean(mutage[int(s[1])])) + '\n'
                else:
                    out += 'NA\n'
            else:
                out += 'Not_mapped\n'
        outfile=open(outpref + '.txt','w')
        outfile.write(out)
        outfile.close()

    def background_archaic_affinity(self, snpinfo, bcffile, outpref, windowsize = 1000):
        """Calculate background archaic affinity for each individual in genome windows."""
        arc_return_dict = {'Chagyrskaya-Phalanx':11, 'AltaiNeandertal':12, 'Vindija33.19':13, 'Denisova':14}
        def parse_snpinfo_anc(self, snpinfo, arc_return_dict):
            infile=open(str(snpinfo))
            lines=infile.readlines()
            infile.close()
            outdict = dict({})
            for i in range(1, len(lines)):
                s=lines[i].strip('\n').strip('\t').split('\t')
                pos = int(s[1])
                ref = str(s[2])
                alt = str(s[3])
                anc = str(s[4])
                dnea = ((s[arc_return_dict['Chagyrskaya-Phalanx']] in ['1','2']) | (s[arc_return_dict['AltaiNeandertal']] in ['1','2']) | (s[arc_return_dict['Vindija33.19']] in ['1','2']))
                nea_missing = (s[arc_return_dict['Chagyrskaya-Phalanx']] == '9') & (s[arc_return_dict['AltaiNeandertal']] == '9') & (s[arc_return_dict['Vindija33.19']] == '9')
                dden = (s[arc_return_dict['Denisova']] in ['1','2'])
                den_missing = (s[arc_return_dict['Denisova']] == '9')
                instrict = True if int(s[-1]) == 1 else False
                if nea_missing or den_missing:
                    outdict["_".join([str(pos), ref, alt])] = ["N", dnea, dden, instrict]
                elif ref == anc:
                    outdict["_".join([str(pos), ref, alt])] = ["keep", dnea, dden, instrict]
                elif alt == anc:
                    outdict["_".join([str(pos), ref, alt])] = ["switch", dnea, dden, instrict]
                elif anc in ['A', 'T', 'C', 'G']:
                    outdict["_".join([str(pos), ref, alt])] = ["other", dnea, dden, instrict]
                else:
                    outdict["_".join([str(pos), ref, alt])] = ["N", dnea, dden, instrict]
            return outdict, pos
        snp_dict, lastsnp = parse_snpinfo_anc(snpinfo, arc_return_dict)
        os.system(
            "bcftools view -v snps " + str(bcffile) + " -Ov -o " + str(outpref) + ".bcftools_temp.vcf"
        )
        infile = open(vcffile, 'r')
        lines = infile.readlines()
        infile.close()
        ind = 0
        while ind < len(lines) and not lines[ind].startswith('#CHROM'):
            ind += 1
        s = lines[ind].strip('\n').strip('\t').split('\t')
        ninds = len(s) - 9  # number of individuals in the VCF
        individuals = []
        for i in range(9, len(s)):
            individuals.append(str(s[i]) + "_hap1")
            individuals.append(str(s[i]) + "_hap2")
        nd00 = np.zeros((int(lastsnp / windowsize) + 1, 2 * ninds))
        nd10 = np.zeros((int(lastsnp / windowsize) + 1, 2 * ninds))
        nd01 = np.zeros((int(lastsnp / windowsize) + 1, 2 * ninds))
        nd11 = np.zeros((int(lastsnp / windowsize) + 1, 2 * ninds))
        als = [nd00, nd10, nd01, nd11]
        ind += 1
        while ind < len(lines):
            s = lines[ind].strip('\n').strip('\t').split('\t')
            pos = "_".join([s[1], s[3], s[4]])
            info = snp_dict[pos]
            idx = int(int(s[1]) / windowsize)
            if info[0] == "N":
                ind += 1
                continue
            elif info[0] == "keep":
                marker = None
                if info[1]:
                    if info[2]:
                        marker = 3
                    else:
                        marker = 1
                else:
                    if info[2]:
                        marker = 2
                    else:
                        marker = 0
                for i in range(9, len(s)):
                    if s[i] == '1|0':
                        als[marker][idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                    elif s[i] == '0|1':
                        als[marker][idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                    elif s[i] == '1|1':
                        als[marker][idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                        als[marker][idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
            elif info[0] == "switch":
                marker = None
                if info[1]:
                    if info[2]:
                        marker = 0
                    else:
                        marker = 2
                else:
                    if info[2]:
                        marker = 1
                    else:
                        marker = 3
                for i in range(9, len(s)):
                    if s[i] == '1|0':
                        als[marker][idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                    elif s[i] == '0|1':
                        als[marker][idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                    elif s[i] == '0|0':
                        als[marker][idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                        als[marker][idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
            else:  # other
                if info[1]:
                    if info[2]:
                        if s[i] == '0|0':
                            nd00[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd00[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '1|0':
                            nd11[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd00[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '0|1':
                            nd00[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd11[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '1|1':
                            nd11[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd11[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                    else:
                        if s[i] == '0|0':
                            nd01[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd01[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '1|0':
                            nd10[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd01[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '0|1':
                            nd01[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd10[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '1|1':
                            nd10[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd10[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                else:
                    if info[2]:
                        if s[i] == '0|0':
                            nd10[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd10[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '1|0':
                            nd01[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd10[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '0|1':
                            nd10[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd01[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '1|1':
                            nd01[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd01[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                    else:
                        if s[i] == '0|0':
                            nd11[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd11[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '1|0':
                            nd00[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd11[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '0|1':
                            nd11[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd00[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
                        elif s[i] == '1|1':
                            nd00[idx, int(int(s[1]) / windowsize), (i - 9) * 2] += 1
                            nd00[idx, int(int(s[1]) / windowsize), (i - 9) * 2 + 1] += 1
            ind += 1
        np.savez_compressed(
            str(outpref) + ".npz",
            nd00=nd00,
            nd10=nd10,
            nd01=nd01,
            nd11=nd11,
            windowsize=windowsize,
            individuals=individuals
        )
        os.remove(outpref + ".bcftools_temp.vcf")

class SUMMARIZE:
    """A class of functions to summarize the results."""

    def append_hmmix_info(self, hmmixfile, summaryfile, outpref, inference = "hmmix", individualID = None):
        """
        Append the HMMIX info to the summary file.
        """
        try:
            hmmix = pd.read_csv(hmmixfile, sep="\s+")
            summary = pd.read_csv(summaryfile, sep="\s+")
        except Exception as e:
            print(f"Error: {e}")
            sys.exit(1)
        if inference == "hmmix":
            hmmix['end'] = hmmix['end'] + 1000
            hmmix = hmmix[hmmix["state"] == "Archaic"]
            hmmix['archaic'] = "Ambiguous"
            hmmix.loc[(hmmix[['AltaiNeandertal', 'Vindija33.19', 'Chagyrskaya-Phalanx']].max(axis=1) > hmmix['Denisova']) & (hmmix['state'] == 'Archaic'), 'archaic'] = 'Neanderthal'
            hmmix.loc[(hmmix[['AltaiNeandertal', 'Vindija33.19', 'Chagyrskaya-Phalanx']].max(axis=1) < hmmix['Denisova']) & (hmmix['state'] == 'Archaic'), 'archaic'] = 'Denisova'
            hmmix['archaic'] = hmmix["mean_prob"].astype(str) + "_" + hmmix['archaic']
            out = ("\t").join(summary.columns) + "\thmmix_start\thmmix_end\thmmix_overlap_length(bp)\tmean_pp\thmmix_assign\n"
        elif inference == "ibdmix":
            if individualID is None:
                print("Please provide the individual ID for ibdmix inference.")
                sys.exit(1)
            hmmix = hmmix[(hmmix["ID"] == individualID) & (hmmix["end"] > hmmix["start"])]
            hmmix['archaic'] = (hmmix['end'] - hmmix['start']).astype(str) + "_" + hmmix['archaic'].astype(str)
            out = ("\t").join(summary.columns) + "\tibdmix_start\tibdmix_end\tibdmix_overlap_length(bp)\tslod\tibdmix_assign\n"
        elif inference == "archie":
            hmmix = pd.read_csv(hmmixfile, sep="\s+", header=None, names=["chrom", "start", "end", "pp", "ID", "overlap"])
            if individualID is None:
                print("Please provide the individual ID for ibdmix inference.")
                sys.exit(1)
            hmmix = hmmix[hmmix["ID"] == individualID]
            hmmix = hmmix[(hmmix['pp'] > 0.9) & (hmmix['overlap'] > 0.7)]
            hmmix['archaic'] = hmmix['pp'].astype(str) + "_ghost"
            out = ("\t").join(summary.columns) + "\tarchie_start\tarchie_end\tarchie_overlap_length(bp)\tpp\tarchie_assign\n"
        s_segs = pybedtools.BedTool(summary[["chromosome", "start", "end"]].to_csv(sep="\t", index=False, header=False), from_string=True)
        h_segs = pybedtools.BedTool(hmmix[["chrom", "start", "end", "archaic"]].to_csv(sep="\t", index=False, header=False), from_string=True)
        overlap = s_segs.intersect(h_segs, wao=True)
        ovl = {}
        for i in range(len(overlap)):
            ol = int(overlap[i][-1])
            if ol > 0:
                key = f"{overlap[i][0]}_{overlap[i][1]}_{overlap[i][2]}"
                if key not in ovl:
                    ovl[key] = []
                ovl[key].append([f"{overlap[i][4]}_{overlap[i][5]}", ol, overlap[i][6]])
        for s in range(len(summary)):
            chrom = summary['chromosome'][s]
            start = summary['start'][s]
            end = summary['end'][s]
            out += ("\t").join(summary.iloc[s].astype(str).tolist())
            if f"{chrom}_{start}_{end}" in ovl:
                ss = ovl[f"{chrom}_{start}_{end}"]
                if len(ss) > 1:
                    hmm_start = ""
                    hmm_end = ""
                    hmm_ol = 0
                    hmm_pp = []
                    hmm_assign = ""
                    for i in range(len(ss)):
                        ss_s1 = ss[i][0].split("_")
                        hmm_start += f"{ss_s1[0]},"
                        hmm_end += f"{ss_s1[1]},"
                        hmm_ol += ss[i][1]
                        ss_s2 = ss[i][2].split("_")
                        hmm_pp.append(float(ss_s2[0]))
                        hmm_assign += f"{ss_s2[1]},"
                    out += f"\t{hmm_start[:-1]}\t{hmm_end[:-1]}\t{hmm_ol}\t{np.max(hmm_pp)}\t{hmm_assign[:-1]}\n"
                else:
                    ss = ss[0]
                    ss_s1 = ss[0].split("_")
                    ss_s2 = ss[2].split("_")
                    out += f"\t{ss_s1[0]}\t{ss_s1[1]}\t{ss[1]}\t{ss_s2[0]}\t{ss_s2[1]}\n"
            else:
                out += "\t-1\t-1\t-1\t-1\t-1\n"
        with open(f"{outpref}.txt", "w") as f:
            f.write(out)
        return

    def parse_snpinfo_anc(self, snpinfo, arc_return_dict):
        """Read in snpinfo file and return a dictionary with ancestral information.

        snpinfo: a str of full file name
        arc_return_dict: a dict indicating position of info for archaics in snpinfo lines.

        return: a dictionary {pos:[following information in the file]}
        """
        infile=open(str(snpinfo))
        lines=infile.readlines()
        infile.close()
        outdict = dict({})
        for i in range(1, len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            pos = int(s[1])
            ref = str(s[2])
            alt = str(s[3])
            anc = str(s[4])
            cpg = True if int(s[10]) == 1 else False
            dnea = ((s[arc_return_dict['Chagyrskaya-Phalanx']] in ['1','2']) | (s[arc_return_dict['AltaiNeandertal']] in ['1','2']) | (s[arc_return_dict['Vindija33.19']] in ['1','2']))
            # dnea = (s[arc_return_dict['Vindija33.19']] in ['1','2'])
            nea_missing = (s[arc_return_dict['Chagyrskaya-Phalanx']] == '9') & (s[arc_return_dict['AltaiNeandertal']] == '9') & (s[arc_return_dict['Vindija33.19']] == '9')
            # nea_missing = (s[arc_return_dict['Vindija33.19']] == '9')
            dden = (s[arc_return_dict['Denisova']] in ['1','2'])
            den_missing = (s[arc_return_dict['Denisova']] == '9')
            freqs = np.array([float(s[15]), float(s[16]), float(s[17])])
            if not "manifesto" in snpinfo:
                aafr = float(s[-3])
                aout = False if float(s[-2]) >= 0.05 else True
                instrict = True if int(s[-1]) == 1 else False
            else:
                aafr = float(s[-4])
                aout = False if float(s[-3]) >= 0.05 else True
                instrict = True if ((int(s[-2]) == 1) & (int(s[-1]) == 1)) else False
            if ref == anc:
                outdict["_".join([str(pos), ref, alt])] = ["keep", dnea, nea_missing, dden, den_missing, aafr, aout, instrict, freqs, cpg]
            elif alt == anc:
                outdict["_".join([str(pos), ref, alt])] = ["switch", dnea, nea_missing, dden, den_missing, aafr, aout, instrict, freqs, cpg]
            elif anc in ['A', 'T', 'C', 'G']:
                outdict["_".join([str(pos), ref, alt])] = ["other", dnea, nea_missing, dden, den_missing, aafr, aout, instrict, freqs, cpg]
            else:
                outdict["_".join([str(pos), ref, alt])] = ["N", dnea, nea_missing, dden, den_missing, aafr, aout, instrict, freqs, cpg]
        return outdict

    def count_snps(self, seg_snps, snp_dict, hap="left"):
        """Count number of derived and archaic snps in the region file.

        seg_snps: str, full name of the region file
        snp_dict: dictionary including snp information

        return: nsnps, nout, ND00, ND10, ND01, ND11, A05, AN01, AD01
        """
        snpcount = {"nsnps":0, "nout":0, "ND00":0, "ND10":0, "ND01":0, "ND11":0, "A05":0, "AN01":0, "AD01":0}
        snpcount_strict = {"nsnps":0, "nout":0, "ND00":0, "ND10":0, "ND01":0, "ND11":0, "A05":0, "AN01":0, "AD01":0}
        dsnps = []
        dsnps_freqs = []
        dsnps_marks = []
        hapdict = {"left":0, "right":1}
        infile=open(seg_snps)
        lines=infile.readlines()
        infile.close()
        for i in range(len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            pos = "_".join([s[0], s[1], s[2]])
            if not pos in snp_dict:
                print(pos)
                print("Error: SNP not found in snp_dict.")
                sys.exit(1)
            if str(s[3]) in ['./.', './1', './0', '0/.', '1/.']: # skip missing sites
                continue
            genos = str(s[3]).split('|')
            geno = int(genos[hapdict[hap]])
            ancinfo, dnea, nea_missing, dden, den_missing, aafr, aout, instrict, freqs, cpg = snp_dict[pos]
            snpcount['A05'] += ((geno > 0 and (aafr < 0.05)) | (geno == 0 and (aafr > 0.95)))
            snpcount['AN01'] += ((geno > 0 and (aafr < 0.05) and dnea) | (geno == 0 and (aafr > 0.95) and not dnea and not nea_missing))
            snpcount['AD01'] += ((geno > 0 and (aafr < 0.05) and dden) | (geno == 0 and (aafr > 0.95) and not dden and not den_missing))
            if instrict:
                snpcount_strict['A05'] += ((geno > 0 and (aafr < 0.05)) | (geno == 0 and (aafr > 0.95)))
                snpcount_strict['AN01'] += ((geno > 0 and (aafr < 0.05) and dnea) | (geno == 0 and (aafr > 0.95) and not dnea and not nea_missing))
                snpcount_strict['AD01'] += ((geno > 0 and (aafr < 0.05) and dden) | (geno == 0 and (aafr > 0.95) and not dden and not den_missing))
            if ancinfo == 'keep':
                if geno > 0:
                    snpcount['nsnps'] += 1
                    snpcount['nout'] += (aout)
                    mark = "other"
                    if not dnea and not nea_missing and not dden and not den_missing:
                        snpcount['ND00'] += 1
                        mark = "ND00"
                    elif dnea and not dden and not den_missing:
                        snpcount['ND10'] += 1
                        mark = "ND10"
                    elif not dnea and not nea_missing and dden:
                        snpcount['ND01'] += 1
                        mark = "ND01"
                    elif dnea and dden:
                        snpcount['ND11'] += 1
                        mark = "ND11"
                    if not cpg:
                        dsnps.append(pos)
                        dsnps_freqs.append(freqs)
                    if instrict:
                        snpcount_strict['nsnps'] += 1
                        snpcount_strict['nout'] += (aout)
                        if not dnea and not nea_missing and not dden and not den_missing:
                            snpcount_strict['ND00'] += 1
                            mark = "ND00_strict"
                        elif dnea and not dden and not den_missing:
                            snpcount_strict['ND10'] += 1
                            mark = "ND10_strict"
                        elif not dnea and not nea_missing and dden:
                            snpcount_strict['ND01'] += 1
                            mark = "ND01_strict"
                        elif dnea and dden:
                            snpcount_strict['ND11'] += 1
                            mark = "ND11_strict"
                    if not cpg:
                        dsnps_marks.append(mark)
            elif ancinfo == 'switch':
                if (1 - geno) > 0:
                    snpcount['nsnps'] += 1
                    snpcount['nout'] += (aout)
                    mark = "other"
                    if dnea and dden:
                        snpcount['ND00'] += 1
                        mark = "ND00"
                    elif not dnea and not nea_missing and dden:
                        snpcount['ND10'] += 1
                        mark = "ND10"
                    elif dnea and not dden and not den_missing:
                        snpcount['ND01'] += 1
                        mark = "ND01"
                    elif not dnea and not nea_missing and not dden and not den_missing:
                        snpcount['ND11'] += 1
                        mark = "ND11"
                    if not cpg:
                        dsnps.append(pos)
                        dsnps_freqs.append(1 - freqs)
                    if instrict:
                        snpcount_strict['nsnps'] += 1
                        snpcount_strict['nout'] += (aout)
                        if dnea and dden:
                            snpcount_strict['ND00'] += 1
                            mark = "ND00_strict"
                        elif not dnea and not nea_missing and dden:
                            snpcount_strict['ND10'] += 1
                            mark = "ND10_strict"
                        elif dnea and not dden and not den_missing:
                            snpcount_strict['ND01'] += 1
                            mark = "ND01_strict"
                        elif not dnea and not nea_missing and not dden and not den_missing:
                            snpcount_strict['ND11'] += 1
                            mark = "ND11_strict"
                    if not cpg:
                        dsnps_marks.append(mark)
            elif ancinfo == 'other':
                snpcount['nsnps'] += 1
                snpcount['nout'] += (aout)
                if not cpg:
                    dsnps.append(pos)
                    freqs = 0*freqs
                    dsnps_freqs.append(freqs)
                    dsnps_marks.append("other")
                if geno > 0:
                    snpcount['ND00'] += (not dnea and not nea_missing and not dden and not den_missing)
                    snpcount['ND10'] += (dnea and not dden and not den_missing)
                    snpcount['ND01'] += (not dnea and not nea_missing and dden)
                    snpcount['ND11'] += (dnea and dden)
                    if instrict:
                        snpcount_strict['nsnps'] += 1
                        snpcount_strict['nout'] += (aout)
                        snpcount_strict['ND00'] += (not dnea and not nea_missing and not dden and not den_missing)
                        snpcount_strict['ND10'] += (dnea and not dden and not den_missing)
                        snpcount_strict['ND01'] += (not dnea and not nea_missing and dden)
                        snpcount_strict['ND11'] += (dnea and dden)
                else:
                    snpcount['ND00'] += (dnea and dden)
                    snpcount['ND10'] += (not dnea and not nea_missing and dden)
                    snpcount['ND01'] += (dnea and not dden and not den_missing)
                    snpcount['ND11'] += (not dnea and not nea_missing and not dden and not den_missing)
                    if instrict:
                        snpcount_strict['nsnps'] += 1
                        snpcount_strict['nout'] += (aout)
                        snpcount_strict['ND00'] += (dnea and dden)
                        snpcount_strict['ND10'] += (not dnea and not nea_missing and dden)
                        snpcount_strict['ND01'] += (dnea and not dden and not den_missing)
                        snpcount_strict['ND11'] += (not dnea and not nea_missing and not dden and not den_missing)
                
        return snpcount, snpcount_strict, np.array(dsnps), np.array(dsnps_freqs), np.array(dsnps_marks)

    def get_regions_bed(self, file):
        """Helper function to read a bed file, concatenate records to bcftools regions string."""
        out = ""
        infile=open(file)
        lines=infile.readlines()
        infile.close() 
        for i in range(len(lines)):
            s = lines[i].strip('\n').strip('\t').split('\t')
            out += s[0] + ':' + s[1] + '-' + s[2]
            if not i == len(lines) - 1:
                out += ','
        return out

    def final_ind_count(self, samplename, summary, hap, snpinfo, bcfpref, outpref):
        """Create summary table for number of mutations per archaic haplotype for each individual.

        samplename: str
        bed: str, bed file prefix of pure archaic regions for the haplotype.
        hap: int, 1 or 2
        snpinfo: str, snpinfo file prefix, should not include chr identifier.
        bcfpref: str, prefix of the original bcffile, should not include chr identifier.

        output: a txt file with archaic segment information and counts of mutations.
        """         
        if os.path.exists(str(outpref) + ".txt") and hap != 2:
            os.remove(str(outpref) + ".txt")
        arc_return_dict = {'Chagyrskaya-Phalanx':11, 'AltaiNeandertal':12, 'Vindija33.19':13, 'Denisova':14}
        infile=open(summary)
        lines=infile.readlines()
        infile.close()
        out = lines[0].strip('\n') + '\tnderived\tnoutgroup\tND00\tND10\tND01\tND11\tA05\tAN01\tAD01\tnderived_strict\tnoutgroup_strict\t'
        out += 'ND00_strict\tND10_strict\tND01_strict\tND11_strict\tA05_strict\tAN01_strict\tAD01_strict\n'
        out1 = lines[0].strip('\n') + "\tdsnps\tdsnps_marks\tDAF_GBR\tDAF_YRI\tDAF_GBRYRI\n"
        cur_chr = lines[1].strip('\n').split('\t')[0]
        snpfile = str(snpinfo) + '.' + cur_chr + '.txt'
        bcffile = str(bcfpref) + cur_chr.strip('chr') + '.bcf'
        snp_dict = self.parse_snpinfo_anc(snpfile, arc_return_dict)
        for i in range(1, len(lines)):
            s=lines[i].strip('\n').strip('\t').split('\t')
            if not cur_chr == str(s[0]):
                if not os.path.exists(str(bcfpref) + str(s[0]).strip('chr') + '.bcf'):
                    continue
                else:
                    cur_chr = str(s[0])
                    snpfile = str(snpinfo) + '.' + cur_chr + '.txt'
                    bcffile = str(bcfpref) + cur_chr.strip('chr') + '.bcf'
                    snp_dict = self.parse_snpinfo_anc(snpfile, arc_return_dict)
            os.system('echo "' + str(s[0]) + "\t" + str(s[1]) + "\t" + str(s[2]) + '" > ' + str(outpref) + str(i) + 'seg.bed')
            reg = self.get_regions_bed(str(outpref) + str(i) + "seg.bed")
            os.system(
                "bcftools view -s " + str(samplename) + " -v snps -r " + str(reg) + " " + str(bcffile) + " | bcftools query -f'[%POS\t%REF\t%ALT\t%GT\n]' > "+ str(outpref) + str(i) + "seg_snps"
            )
            snpcount, snpcount_strict, dsnps, dsnps_freqs, dsnps_marks = self.count_snps(str(outpref) + str(i) + "seg_snps", snp_dict, hap)
            out += lines[i].strip('\n') + '\t' + str(snpcount['nsnps']) + '\t' + str(snpcount['nout']) + '\t' + str(snpcount['ND00']) + '\t' 
            out += str(snpcount['ND10']) + '\t' + str(snpcount['ND01']) + '\t' + str(snpcount['ND11']) + '\t' + str(snpcount['A05']) + '\t' 
            out += str(snpcount['AN01']) + '\t' + str(snpcount['AD01']) + '\t' + str(snpcount_strict['nsnps']) + '\t' + str(snpcount_strict['nout']) + '\t' 
            out += str(snpcount_strict['ND00']) + '\t' + str(snpcount_strict['ND10']) + '\t' + str(snpcount_strict['ND01']) + '\t' + str(snpcount_strict['ND11']) + '\t'
            out += str(snpcount_strict['A05']) + '\t' + str(snpcount_strict['AN01'])  + '\t'  + str(snpcount_strict['AD01'])  + "\n"
            dsnps_freqs = np.round(dsnps_freqs, 3)
            if len(dsnps) > 0:
                out1 += lines[i].strip('\n') + '\t' + ','.join(dsnps) + '\t'
                out1 += ','.join(dsnps_marks) + '\t'
                out1 += ','.join([str(i) for i in dsnps_freqs[:, 0]]) + '\t'
                out1 += ','.join([str(i) for i in dsnps_freqs[:, 1]]) + '\t'
                out1 += ','.join([str(i) for i in dsnps_freqs[:, 2]]) + '\n'
            else:
                out1 += lines[i].strip('\n') + '\t' + 'NA\tNA\tNA\tNA\tNA\n'
            os.remove(str(outpref) + str(i) + 'seg.bed')
            os.remove(str(outpref) + str(i) + "seg_snps")
        outfile=open(outpref + '.txt','a')
        outfile.write(out)
        outfile.close()
        outfile=open(outpref + '_DAF.txt','w')
        outfile.write(out1)
        outfile.close()

    def append_t1_t2(self, npzpref, summary, snpinfo, func, outpref):
        mutage = {}
        infile = open(snpinfo)
        lines = infile.readlines()
        infile.close()
        for i in range(1, len(lines)):
            s = lines[i].strip('\n').strip('\t').split('\t')
            pos = int(s[1])
            if not s[-1] in ['nan', 'Not_mapped', 'NA']:
                ag = float(s[-1])
                mutage[pos] = ag
        with np.load(f"{npzpref}.npz") as data:
            t1 = data["t1s"]
            t2 = data["t2s"]
            nleaves = data["nleaves"]
            treespan_phy = data["treespan_phy"]
        windowsize = treespan_phy[0][1] - treespan_phy[0][0]
        if func == "mean":
            func = np.mean
        elif func == "median":
            func = np.median
        else:
            sys.exit(f"Unrecognized function {func}")
        # t1 = func(t1, axis = 0)
        # t2 = func(t2, axis = 0)
        # nleaves = np.nanmean(nleaves, axis = 0)
        infile=open(summary)
        lines=infile.readlines()
        infile.close()
        out = lines[0].strip('\n') + '\tt1s\tt2s\tn_leaves\tmutages\tbranch_mark\n'
        for i in range(1, len(lines)):
            s = lines[i].strip('\n').strip('\t').split('\t')
            st = int(int(s[1]) / windowsize)
            ed = int(int(s[2]) / windowsize)
            t1_val = ",".join(t1[st:ed].astype('str'))
            t2_val = ",".join(t2[st:ed].astype('str'))
            nlv_val = ",".join(nleaves[st:ed].astype('str'))
            out += lines[i].strip('\n') + '\t' + str(t1_val) + '\t' + str(t2_val) + '\t' + str(nlv_val) + '\t'
            muts = s[-5].strip('\n').strip('\t').split(',')
            mks = []
            ags = []
            for m in muts:
                pos = int(m.split('_')[0])
                if pos in mutage:
                    ags.append(mutage[pos])
                    if mutage[pos] > t1[int(pos / windowsize)] and mutage[pos] < t2[int(pos / windowsize)]:
                        mks.append("on")
                    elif mutage[pos] <= t1[int(pos / windowsize)]:
                        mks.append("below")
                    elif mutage[pos] >= t2[int(pos / windowsize)]:
                        mks.append("above")
                else:
                    ags.append("NA")
                    mks.append("NA")
            out += "\t" + ",".join([str(i) for i in ags]) + "\t" + ",".join(mks) + '\n'
        outfile=open(outpref + '.txt','w')
        outfile.write(out)
        outfile.close()

    def append_seg_freq(self, npzpref, indlist, state_arc, outpref, chrom):
        """Append segment frequency in defined population to the summary file."""
        if state_arc == "NEA":
            sn = "nea_states"
        elif state_arc == "DEN":
            sn = "den_states"
        elif state_arc == "Ghost":
            sn = "ghost_states"
        else:
            sys.exit(f"Unrecognized state_arc {state_arc}")
        with np.load(f"{npzpref}{indlist[0]}.{chrom}.xss.npz") as data:
            treespan_phy = data["treespan_phy"]
        states = np.zeros((len(indlist), treespan_phy.shape[0]))
        with np.load(f"{npzpref}{indlist[0]}.{chrom}.xss.npz") as data:
            tsp = data["treespan_phy"]
        states = np.zeros((len(indlist), tsp.shape[0]))
        for idx in range(len(indlist)):
            with np.load(f"{npzpref}{indlist[idx]}.{chrom}.xss.npz") as data:
                try:
                    s = data[sn]
                    states[idx] = s
                except:
                    print(f"Error: {indlist[idx]} does not have {sn} in {npzpref}{indlist[idx]}.{chrom}.xss.npz")
                    continue
        np.savez_compressed(f"{outpref}.npz", states=states, treespan_phy=treespan_phy)

    def append_bscore_recombrate(self, popcountfile, bscore, recombrate):
        """Append Bscore and recombination rate for the region."""
        infile = open(popcountfile, 'r')
        lines = infile.readlines()
        infile.close()
        b2 = pybedtools.BedTool(bscore)
        b3 = pybedtools.BedTool(recombrate)
        out = lines[0].strip('\n') + '\tbscore\trecombrate\n'
        for i in range(1, len(lines)):
            s = lines[i].strip('\n').split('\t')
            bedstring = str(s[0]) + '\t' + str(s[1]) + '\t' + str(s[2])
            b1 = pybedtools.BedTool(bedstring, from_string=True)
            ib1 = b2.intersect(b1)
            bsc = np.sum([int(ib1[i][3]) * (int(ib1[i][2]) - int(ib1[i][1])) for i in range(len(ib1))]) / (int(s[2]) - int(s[1]))
            out += lines[i].strip('\n') + '\t' + str(bsc) + '\t'
            ib1 = b3.intersect(b1)
            rec = np.sum([float(ib1[i][3]) * (int(ib1[i][2]) - int(ib1[i][1])) for i in range(len(ib1))]) / (int(s[2]) - int(s[1]))
            out += str(float(rec)) + '\n'
        outfile = open(popcountfile, 'w')
        outfile.write(out)
        outfile.close()

    def bscore_recombrate_windows(self, npzfile, bscore, recombrate, chrom):
        """Get per 1000bp Bscore and recombination rate."""
        d = np.load(npzfile)
        data = {k: d[k] for k in d.files} 
        treespan_phy = data["treespan_phy"]
        treespan_cM = GhostProductHmm().add_recombination_map(treespan_phy, recombrate)
        recrate = (treespan_cM[:, 1] - treespan_cM[:, 0]) / (1000 * 1e-6)
        bstring = ""
        for i in range(treespan_phy.shape[0]):
            bstring += f"{chrom}\t{treespan_phy[i, 0]}\t{treespan_phy[i, 1]}\n"
        b1 = pybedtools.BedTool(bstring, from_string=True).sort()
        b2 = pybedtools.BedTool(bscore).sort()
        maped = b1.map(b2, c=4, o='mean', null=0)
        bsc = np.array([float(field[3]) for field in maped])
        data["bscore"] = bsc
        data["recombrate"] = recrate
        np.savez_compressed(npzfile, **data)



# def background_archaic_affinity(snpinfo, bcffile, outpref, windowsize = 1000, strictmask = False):
#     """Calculate background archaic affinity for each individual in genome windows."""
#     arc_return_dict = {'Chagyrskaya-Phalanx':11, 'AltaiNeandertal':12, 'Vindija33.19':13, 'Denisova':14}
#     def parse_snpinfo_anc(snpinfo, arc_return_dict):
#         infile=open(str(snpinfo))
#         lines=infile.readlines()
#         infile.close()
#         outdict = dict({})
#         for i in range(1, len(lines)):
#             s=lines[i].strip('\n').strip('\t').split('\t')
#             pos = int(s[1])
#             ref = str(s[2])
#             alt = str(s[3])
#             anc = str(s[4])
#             dnea = ((s[arc_return_dict['Chagyrskaya-Phalanx']] in ['1','2']) | (s[arc_return_dict['AltaiNeandertal']] in ['1','2']) | (s[arc_return_dict['Vindija33.19']] in ['1','2']))
#             nea_missing = (s[arc_return_dict['Chagyrskaya-Phalanx']] == '9') & (s[arc_return_dict['AltaiNeandertal']] == '9') & (s[arc_return_dict['Vindija33.19']] == '9')
#             dden = (s[arc_return_dict['Denisova']] in ['1','2'])
#             den_missing = (s[arc_return_dict['Denisova']] == '9')
#             instrict = True if int(s[-1]) == 1 else False
#             if nea_missing or den_missing:
#                 outdict["_".join([str(pos), ref, alt])] = ["N", dnea, dden, instrict]
#             elif ref == anc:
#                 outdict["_".join([str(pos), ref, alt])] = ["keep", dnea, dden, instrict]
#             elif alt == anc:
#                 outdict["_".join([str(pos), ref, alt])] = ["switch", dnea, dden, instrict]
#             elif anc in ['A', 'T', 'C', 'G']:
#                 outdict["_".join([str(pos), ref, alt])] = ["other", dnea, dden, instrict]
#             else:
#                 outdict["_".join([str(pos), ref, alt])] = ["N", dnea, dden, instrict]
#         return outdict, pos
#     snp_dict, lastsnp = parse_snpinfo_anc(snpinfo, arc_return_dict)
#     os.system(
#         "bcftools view -v snps " + str(bcffile) + " -Ov -o " + str(outpref) + ".bcftools_temp.vcf"
#     )
#     infile = open(str(outpref) + ".bcftools_temp.vcf", 'r')
#     lines = infile.readlines()
#     infile.close()
#     ind = 0
#     while ind < len(lines) and not lines[ind].startswith('#CHROM'):
#         ind += 1
#     s = lines[ind].strip('\n').strip('\t').split('\t')
#     ninds = len(s) - 9  # number of individuals in the VCF
#     individuals = []
#     for i in range(9, len(s)):
#         individuals.append(str(s[i]) + "_hap1")
#         individuals.append(str(s[i]) + "_hap2")
#     nd00 = np.zeros((int(lastsnp / windowsize) + 1, 2 * ninds))
#     nd10 = np.zeros((int(lastsnp / windowsize) + 1, 2 * ninds))
#     nd01 = np.zeros((int(lastsnp / windowsize) + 1, 2 * ninds))
#     nd11 = np.zeros((int(lastsnp / windowsize) + 1, 2 * ninds))
#     als = [nd00, nd10, nd01, nd11]
#     ind += 1
#     while ind < len(lines):
#         s = lines[ind].strip('\n').strip('\t').split('\t')
#         pos = "_".join([s[1], s[3], s[4]])
#         info = snp_dict[pos]
#         idx = int(int(s[1]) / windowsize)
#         if info[0] == "N":
#             ind += 1
#             continue
#         if strictmask and not info[3]:
#             ind += 1
#             continue
#         elif info[0] == "keep":
#             marker = None
#             if info[1]:
#                 if info[2]:
#                     marker = 3
#                 else:
#                     marker = 1
#             else:
#                 if info[2]:
#                     marker = 2
#                 else:
#                     marker = 0
#             for i in range(9, len(s)):
#                 if s[i] == '1|0':
#                     als[marker][idx, (i - 9) * 2] += 1
#                 elif s[i] == '0|1':
#                     als[marker][idx, (i - 9) * 2 + 1] += 1
#                 elif s[i] == '1|1':
#                     als[marker][idx, (i - 9) * 2] += 1
#                     als[marker][idx, (i - 9) * 2 + 1] += 1
#         elif info[0] == "switch":
#             marker = None
#             if info[1]:
#                 if info[2]:
#                     marker = 0
#                 else:
#                     marker = 2
#             else:
#                 if info[2]:
#                     marker = 1
#                 else:
#                     marker = 3
#             for i in range(9, len(s)):
#                 if s[i] == '1|0':
#                     als[marker][idx, (i - 9) * 2 + 1] += 1
#                 elif s[i] == '0|1':
#                     als[marker][idx, (i - 9) * 2] += 1
#                 elif s[i] == '0|0':
#                     als[marker][idx, (i - 9) * 2] += 1
#                     als[marker][idx, (i - 9) * 2 + 1] += 1
#         else:  # other
#             if info[1]:
#                 if info[2]:
#                     if s[i] == '0|0':
#                         nd00[idx, (i - 9) * 2] += 1
#                         nd00[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '1|0':
#                         nd11[idx, (i - 9) * 2] += 1
#                         nd00[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '0|1':
#                         nd00[idx, (i - 9) * 2] += 1
#                         nd11[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '1|1':
#                         nd11[idx, (i - 9) * 2] += 1
#                         nd11[idx, (i - 9) * 2 + 1] += 1
#                 else:
#                     if s[i] == '0|0':
#                         nd01[idx, (i - 9) * 2] += 1
#                         nd01[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '1|0':
#                         nd10[idx, (i - 9) * 2] += 1
#                         nd01[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '0|1':
#                         nd01[idx, (i - 9) * 2] += 1
#                         nd10[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '1|1':
#                         nd10[idx, (i - 9) * 2] += 1
#                         nd10[idx, (i - 9) * 2 + 1] += 1
#             else:
#                 if info[2]:
#                     if s[i] == '0|0':
#                         nd10[idx, (i - 9) * 2] += 1
#                         nd10[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '1|0':
#                         nd01[idx, (i - 9) * 2] += 1
#                         nd10[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '0|1':
#                         nd10[idx, (i - 9) * 2] += 1
#                         nd01[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '1|1':
#                         nd01[idx, (i - 9) * 2] += 1
#                         nd01[idx, (i - 9) * 2 + 1] += 1
#                 else:
#                     if s[i] == '0|0':
#                         nd11[idx, (i - 9) * 2] += 1
#                         nd11[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '1|0':
#                         nd00[idx, (i - 9) * 2] += 1
#                         nd11[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '0|1':
#                         nd11[idx, (i - 9) * 2] += 1
#                         nd00[idx, (i - 9) * 2 + 1] += 1
#                     elif s[i] == '1|1':
#                         nd00[idx, (i - 9) * 2] += 1
#                         nd00[idx, (i - 9) * 2 + 1] += 1
#         ind += 1
#     np.savez_compressed(
#         str(outpref) + ".npz",
#         nd00=nd00,
#         nd10=nd10,
#         nd01=nd01,
#         nd11=nd11,
#         windowsize=windowsize,
#         individuals=individuals
#     )
#     os.remove(outpref + ".bcftools_temp.vcf")


# snpinfo = "results/realdata/1000g_hg38_2022/snpinfo/1000g_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.chr1.txt"
# bcffile = "results/realdata/1000g_hg38_2022/vcf_files/1000g_hg38_chr1.bcf"
# outpref = "results/realdata/1000g_hg38_2022/snpinfo/chr1.background_archaic_affinity.strictmask"
# background_archaic_affinity(snpinfo, bcffile, outpref, windowsize = 1000, strictmask=True)

