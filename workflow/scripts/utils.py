"""Utils for running simulation-based analysis workflow."""
import json
import numpy as np
import pandas as pd
import tskit
import tszip
from tqdm import tqdm
from pathlib import Path
import sys
from trace.trace import TRACE
from utils import ARG_utils, Performance_utils
import os
import pybedtools
from pyfaidx import Fasta


class Analysis_workflow_utils:
    """A class of functions used in analysis workflow."""
    
    def run_hmm(self, ncoal, treespan, intro_prop, seed):
        """A helper function to run TRACE."""
        hmm = TRACE()
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
        ahmm = TRACE()
        ancoal, at1s, at2s, atreespan, anleaves = ahmm.prepare_data_tmrca(ts = target_ts, ind = ind, t_archaic=t_archaic)
        n_gamma, aoutdict, atreespan = self.run_hmm(ncoal = ancoal, treespan = atreespan, intro_prop = intro_prop, seed = seed)
        a_param = pd.DataFrame.from_dict(aoutdict).to_numpy()

        # init hmm for outgroup population
        fhmm = TRACE()
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
            hmm = TRACE()
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

