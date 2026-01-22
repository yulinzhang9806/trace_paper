"""Utility functions for ARGweaver related utils."""

import math
import os
import random
from random import sample

import ete3
import numpy as np
import pandas as pd
import tskit
from ete3 import Tree
from numpy.random import choice
from scipy.stats import gamma, poisson
# import tszip
# import json


class ARGweaver_to_ts:
    """Methods used to convert ARGweaver output smc file to tskit tree sequence.

    Main Author: Yun Deng (Minor edits by Yulin Zhang).

    Notice:
    1. ARGweaver would produce loops in tree sequences, which are not acceptable in tskit tree format. We discard recombination nodes here and treat all internal nodes (nodes except for leaves / sample nodes) as independent new nodes in different trees to avoid loops. This breaks the continuity of branches and internal nodes in the tree sequence, but would not affect our direct usage of TMRCAs or branch lengths in marginal trees.
    2. ARGweaver uses discrete time setting, which would lead to branches with length 0 in marginal trees. To make the tree structure acceptable to tskit format, we introduce slight perturbation to these branches so that to make the branch length non-zero. The output tree structure in tskit would possibly non-binary due to these extremely short branches.
    """

    def __init__(
        self,
    ):
        """Initialize the ARGweaver_to_ts class."""
        self.ts = None
        self.ts_sample_names = None

    def get_time_map_helper(self, tree, time_map):
        """Help (er) function for get_time_map."""
        for c in tree.children:
            time_map[c.name] = time_map[tree.name] - c.dist
            self.get_time_map_helper(c, time_map)

    def get_time_map(self, tree):
        """Get time for each node in the tree."""
        time_map = {}
        time_map[tree.name] = 0
        self.get_time_map_helper(tree, time_map)
        min_time = 0
        for n in time_map:
            min_time = min(min_time, time_map[n])
        for n in time_map:
            time_map[n] = time_map[n] - min_time
        return time_map

    def get_sample_nodes(self, tree, sample_name):
        """Return two dictionaries.

        sample_nodes: argweaver nodes -> ts nodes.
        ts_sample_names: ts nodes -> sample names.
        """
        sample_nodes = {}
        ts_sample_names = {}
        count = 0
        for n in tree.traverse("preorder"):
            if n.is_leaf():
                sample_nodes[n.name] = count
                count += 1
        for k in sample_nodes:
            ts_sample_names[sample_nodes[k]] = sample_name[int(k)]
        return sample_nodes, ts_sample_names

    def write_sample_nodes(self, tree, tables):
        """Write nodes info to ts tables."""
        time_map = self.get_time_map(tree)
        for n in tree.traverse("preorder"):
            if n.is_leaf():
                tables.nodes.add_row(flags=tskit.NODE_IS_SAMPLE, time=time_map[n.name])
        return

    def get_perturbed_time_map(self, tree):
        """Introduce perturbation to branches with 0 length."""
        time_map = self.get_time_map(tree)
        sort = []
        for n in tree.traverse("preorder"):
            sort.append(n)
        m = len(sort)
        for i in range(len(sort)):
            n = sort[i]
            if len(n.children) > 0:
                time_map[n.name] = time_map[n.name] + (m - i) * 0.1
        return time_map

    def update_table(self, tables, tree, sample_nodes, left, right):
        """Update tables."""
        index_map = {}
        time_map = self.get_perturbed_time_map(tree)
        for x in tree.traverse():
            if x.name not in sample_nodes:
                index_map[x.name] = tables.nodes.num_rows
                tables.nodes.add_row(time=time_map[x.name])
            else:
                index_map[x.name] = sample_nodes[x.name]
        for x in tree.traverse():
            for y in x.children:
                assert time_map[y.name] < time_map[x.name]
                tables.edges.add_row(left, right, index_map[x.name], index_map[y.name])
        return

    def get_sample_name(self, filename):
        """Return a dictionary with argweaver node -> sample names."""
        infile = open(filename)
        lines = infile.readlines()
        infile.close()
        s = lines[0].strip("\n").strip("\t").split("\t")
        sample_name = dict({})
        for i in range(1, len(s)):
            sample_name[i - 1] = str(s[i])
        return sample_name

    def read_smc(self, filename):
        """Wrap up function to read smc file and get ts tree sequence."""
        sample_name = self.get_sample_name(filename)
        df = pd.read_csv(
            filename,
            skiprows=lambda x: x % 2 == 1 or x == 0,
            delimiter="\t",
            header=None,
        )
        sl = df.iloc[-1, 2]
        tables = tables = tskit.TableCollection(sequence_length=sl)
        init_tree = Tree(df.iloc[0, 3], format=1)
        sample_nodes, ts_sample_names = self.get_sample_nodes(init_tree, sample_name)
        self.write_sample_nodes(init_tree, tables)
        n = df.shape[0]
        for i in range(n):
            tree = Tree(df.iloc[i, 3], format=1)
            left = df.iloc[i, 1] - df.iloc[0, 1]
            right = df.iloc[i, 2]  # - df.iloc[0, 1] + 1
            self.update_table(tables, tree, sample_nodes, left, right)
        tables.sort()
        ts = tables.tree_sequence()
        self.ts = ts
        self.ts_sample_names = ts_sample_names
        return ts, ts_sample_names


# class ARGweaver_topology_utils:
#     """Methods used to directly extract marginal tree information from ARGweaver output SMC tree sequence."""

#     def __init__(
#         self,
#     ):
#         """Initialize the ARGweaver_util class."""
#         self.tree_seq = None
#         self.tree_dist = None
#         self.sample_names = None
#         self.tmrca_seq = None

#     def read_smc(self, filename):
#         """Read a smc tree sequence file.

#         sample_names: a dictionary with argweaver node -> sample names.
#         tree_seq: a 1-dim array, each entry is newick format of a marginal tree.
#         tree_dist: an array, each entry recording the start and end of a marginal tree.
#         """
#         self.sample_names = ARGweaver_to_ts().get_sample_name(filename)
#         df = pd.read_csv(
#             filename,
#             skiprows=lambda x: x % 2 == 1 or x == 0,
#             delimiter="\t",
#             header=None,
#         )
#         tree_seq = []
#         tree_dist = []
#         for i in range(len(df)):
#             dist = [df.iloc[i, 1], df.iloc[i, 2]]
#             tree_seq.append(df.iloc[i, 3])
#             tree_dist.append(dist)
#         self.tree_seq = np.array(tree_seq)
#         self.tree_dist = np.array(tree_dist)
#         return self.sample_names, tree_seq, tree_dist

#     def extract_tmrca(self, i, tree):
#         """Extract all tmrcas of the target individual verses all other individuals in the population for the current tree.

#         ind: the argweaver node id.
#         tmrca: an array of tmrca, shape = (1, len(samples) - 1)

#         """
#         cur_tree = Tree(tree, format=1)
#         tmrca = np.array(
#             [
#                 [
#                     cur_tree.get_distance(str(i), str(j))
#                     for j in self.sample_names
#                     if i != j
#                 ]
#             ]
#         )
#         return tmrca

#     def extract_tmrca_seq(self, ind, sampleName=True, f=None):
#         """Extract all tmrcas of the target individual verses all other individuals in the population for the tree sequence.

#         ind: the argweaver node (sampleName = False) or the sample name (sampleName = True).
#         sampleName: see above.
#         tmrca_seq: an array including all tmrcas between the target individual vs. all other individuals in the population (or summary stats f of them) in all marginal trees.

#         """
#         if f is None:
#             tmrca_seq = np.zeros([len(self.tree_seq, len(self.sample_name) - 1)])
#         else:
#             tmrca_seq = np.zeros([1, len(self.tree_seq)])
#         if sampleName:
#             assert ind in self.sample_names.values()
#             node_id = list(self.sample_names.keys())[
#                 list(self.sample_names.values()).index(ind)
#             ]
#         else:
#             assert ind in self.sample_names.keys()
#             node_id = ind
#         for i in range(len(self.tree_seq)):
#             tmrca = self.extract_tmrca(node_id, self.tree_seq[i])
#             if f is None:
#                 tmrca_seq[i] = tmrca
#             else:
#                 tmrca_seq[0, i] = f(tmrca)
#         self.tmrca_seq = tmrca_seq
#         return tmrca_seq

#     def extract_coal_events(self, ind, sampleName=True, t1=100, t2=1000):
#         """Method to extract the number of coalescent events features from ARGweaver trees.
#         Arguments:

#         ind: the argweaver node (sampleName = False) or the sample name (sampleName = True), int.
#         sampleName: see above.
#         - t1: time in generations for lower bound (tadmix)
#         - t2: time in generations for upper bound (tarchaic)

#         """
#         if sampleName:
#             assert ind in self.sample_names.values()
#             node_id = list(self.sample_names.keys())[
#                 list(self.sample_names.values()).index(ind)
#             ]
#         else:
#             assert ind in self.sample_names.keys()
#             node_id = ind
#         for i in range(len(self.tree_seq)):
#             cur_tree = Tree(self.tree_seq[i], format=1)
#             t = ind
#             while not t.is_


# ########## Work in progress ########
# # tree node and node name are different, solve this by traversing through the leaves.


#     # def get_isfs(self, ind, sampleName=True):
#     #     """Get iSFS for the given individual.

#     #     Need working.
#     #     """
#     #     return


class ARGweaver_related_utils:
    """Utility to create ARGweaver related files."""

    def __init__(
        self,
        total_sample=2000,
        sample_step=10,
    ):
        """Initialize the ARGweaver_util class."""
        self.total_sample = total_sample
        self.sample_step = sample_step

    def getSitesInput(self, input_size, vcffile_path, outpref, msprime=True):
        """Create ARGweaver input from vcf file."""
        infile = open(vcffile_path)
        lines = infile.readlines()
        infile.close()
        s = lines[-1].strip("\n").strip("\t").split("\t")
        chrom_id = str(s[0])
        total_len = int(s[1])
        split_num = math.ceil(total_len / input_size)
        ind = 0
        while lines[ind].startswith("##"):
            ind += 1
        j = ind + 1
        for i in range(split_num):
            output = "NAMES\t"
            s = lines[ind].strip("\n").strip("\t").split("\t")
            if not msprime:
                for k in range(9, len(s)):
                    output += str(s[k]) + "_hap1\t"
                    output += str(s[k]) + "_hap2\t"
            else:
                for k in range(9, len(s)):
                    output += str((k - 9) * 2) + "\t" + str((k - 9) * 2 + 1) + "\t"
            output = output.strip("\t")
            output += (
                "\nREGION\t"
                + "chr"
                + str(chrom_id)
                + "\t"
                + str(int(i * input_size + 1))
                + "\t"
                + str(int((i + 1) * input_size))
                + "\n"
            )
            while j < len(lines):
                s = lines[j].strip("\n").strip("\t").split("\t")
                ref = s[3]
                alt = s[4]
                if (
                    int(s[1]) >= i * input_size + 1
                    and int(s[1]) <= (i + 1) * input_size
                ):
                    output += str(s[1]) + "\t"
                    for t in range(9, len(s)):
                        geno = s[t].strip("\t").split("|")
                        for g in geno:
                            if int(g) == 0:
                                output += ref
                            else:
                                output += alt
                    output += "\n"
                    j += 1
                else:
                    break
            outfile = open(str(outpref) + "." + str(i) + ".sites", "w")
            outfile.write(output)
            outfile.close()

    def subsampleVCF(self, sample_file, fix_sample, num_sample, outfile, seed):
        """Generate a subset of samples.

        Note, num_sample don`t need to -1 for fix_sample spot.
        """
        infile = open(sample_file)
        lines = infile.readlines()
        infile.close()
        ss = []
        ind = 0
        while ind < len(lines):
            s = lines[ind].strip("\n").strip("\t").split("\t")
            ss.append(s[0])
            ind += 1
        output = str(fix_sample) + "\n"
        random.seed(seed)
        sub = sample(ss, num_sample - 1)
        while fix_sample in sub:
            sub = sample(ss, num_sample - 1)
        for i in sub:
            output += str(i) + "\n"
        outfile = open(outfile, "w")
        outfile.write(output)
        outfile.close()

    def samplePosterior(self, prefix, dimiss_num, nsamp, seed, replace=False):
        """Remove useless posterior samples. Sample certain number of tree sequences from posterior. Return a numpy array with numbers.

        dismiss_num: sample the result starting from output of this number in posterior tree sequences.
        nsamp: number of posterior samples need to be returned.
        sample_step: the sampling step of ARGweaver.
        """
        assert dimiss_num < self.total_sample
        s_set = np.arange(0, self.total_sample, 10)
        s_set = s_set[s_set > dimiss_num]
        if nsamp > len(s_set):
            print("Number of samples larger than sample set.")
            if replace is False:
                print("Error: Sampling without replacement.")
                return
        for j in np.arange(0, dimiss_num, 10):
            if os.path.exists(str(prefix) + "." + str(j) + ".smc.gz"):
                os.remove(str(prefix) + "." + str(j) + ".smc.gz")
        df = pd.read_csv(str(prefix) + ".stats", sep="\s+")
        df = df[df["iter"].isin(s_set)]
        # weight=dict()
        # for i in range(len(df['joint'].unique())):
        #     weight[sorted(df['joint'].unique())[i]] = sorted((df['joint'] / sum(df['joint'])).unique())[i]
        # df['weight']=df['joint'].map(weight)
        # if replace:
        #     out = df.sample(
        #         n=nsamp, weights="weight", replace=True, random_state=seed
        #     )["iter"].to_numpy()
        # else:
        #     out = df.sample(n=nsamp, weights="weight", random_state=seed)[
        #         "iter"
        #     ].to_numpy()
        out_arr = np.zeros(shape = (2, nsamp))
        out_arr[0] = df.sort_values(by="joint", ascending=False)["iter"].astype('int').to_numpy()[:nsamp]
        out_arr[1] = df.sort_values(by="joint", ascending=False)["joint"].to_numpy()[:nsamp]
        return out_arr

    # def get_n_coal_ind(self, ind, prefix, nsamp, t1, t2, n, windowsize=1000, genomesize=10e6):
    #     """Get n_coal by using mean TMRCA across posterior ARGweaver samples for each individual."""
        
    #     out_coal = []
    #     for pos in range(0, genomesize, windowsize):
    #         p = pos + genomesize / 2
    #         temp_tmrca = np.zeros(shape = (nsamp, n))
    #         for i in range(nsamp):
    #             ts = tszip.decompress(str(prefix) + "." + str(i) + ".tsz")
    #             with open(
    #                 str(prefix) + "." + str(i) + ".sample.json", "r"
    #             ) as f:
    #                 ts_to_arg = json.load(f)
    #             arg_to_ts = dict((int(v), int(k)) for k, v in ts_to_arg.items())
    #             tree = ts.at(p)
    #             for s in range(n):
    #                 temp_tmrca[i][s] = tree.tmrca(arg_to_ts[ind], arg_to_ts[s])
    #         mean_tmrca = np.mean(temp_tmrca, axis = 0)
    #         n_coal = 
    #         out_coal.append()

        
        
        
        
    #     treespan=dict()
    #     for i in range(genomesize / windowsize):
    #         treespan[i] = [i * windowsize, (i + 1) * windowsize - 1]
        

    def argweaver_avg_pairwise_times(self, s, k1, k2, filepath, fsample):
        """Avg pairwise tmrca from argweaver output.
        
        fsample: list
        """
        ts = tszip.decompress(filepath + '.0.tsz')
        out = np.zeros(shape=(len(fsample), int(ts.sequence_length / s + (ts.sequence_length % s > 0))))
        for i in range(len(fsample)):
            ts = tszip.decompress(filepath + "." + str(fsample[i]) + ".tsz")
            with open(
                str(filepath) + "." + str(fsample[i]) + ".sample.json", "r"
            ) as f:
                ts_to_arg = json.load(f)
            arg_to_ts = dict((int(v), int(k)) for k, v in ts_to_arg.items())
            windows, div = self.get_pairwise_times(ts, s, arg_to_ts[k1], arg_to_ts[k2])
            out[i] = div
        return windows, np.mean(out, axis=0)

    def run_hmm_avg_trees(
        self, 
        inds,
        s, 
        filepath,
        fsample,
        t1,
        t2,
        t2_low,
        njob,
        argweaver=True,
    ):
        """Run hmm on avg posterior trees."""

        # extract avg tmrcas across fsample trees
        def get_pair_tmrcas(ts, filepath, fsample, s, ind, argweaver):
            other_samples = np.delete(np.arange(ts.num_samples), np.where(np.arange(ts.num_samples) == ind))
            xss = np.zeros(shape = (int(ts.sequence_length / s + (ts.sequence_length % s > 0)), ts.num_samples - 1))
            for i in range(len(other_samples)):
                if argweaver:
                    windows, tmrcas = self.argweaver_avg_pairwise_times(s, ind, other_samples[i], filepath, fsample)
                else:
                    windows, tmrcas = self.singer_avg_pairwise_times(s, ind, other_samples[i], filepath, fsample)
                xss[:, i] = tmrcas
            return xss

        # run hmm on defined tmrcas
        if argweaver:
            ts = tszip.decompress(str(filepath) + ".0.tsz")
        else:
            ts = tskit.load(str(filepath) + "_0.trees")
        hmm = GhostProductHmm()
        hmm.add_tree_sequence(ts)
        hmm.treespan = dict({})
        windows = np.arange(0, ts.sequence_length, s)
        windows = np.append(windows, ts.sequence_length)
        hmm.m = len(windows) - 1
        hmm.pos = np.zeros(len(windows) - 1)
        for i in range(len(windows) - 1):
            hmm.pos[i] = (windows[i] + windows[i + 1]) / 2.0
            hmm.treespan[i] = np.array([windows[i], windows[i + 1]])
        hmm.set_constant_recomb()
        hmm.xss = np.zeros(shape = (ts.num_samples, int(ts.sequence_length / s + (ts.sequence_length % s > 0)), ts.num_samples - 1))
        hmm.emissions = np.zeros(shape=(ts.num_samples, 2, int(ts.sequence_length / s + (ts.sequence_length % s > 0))))
        outputs = Parallel(n_jobs=njob)(delayed(get_pair_tmrcas)(ts = ts, filepath = filepath, fsample = fsample, s = s, ind = ind, argweaver = argweaver) for ind in range(ts.num_samples))
        for ind in range(ts.num_samples):
            hmm.xss[ind] = outputs[ind]
        hmm.est_null_kde(i = inds)
        cp, prop = hmm.init_admix_gamma_params(t_admix = t1, t_archaic = t2, t_archaic_low = t2_low)
        res_dict = hmm.train(inds = inds, niter = 180, g_prop=prop, njobs = njob)
        gammas, alphas, betas = hmm.decode(inds = inds, njobs = njob)
        return (
            np.exp(gammas),
            res_dict,
            hmm.xss,
            hmm.treespan,
            t2_low, 
            cp, 
            prop
        )

    def record_hmm_results_avg(
        self,
        inds,
        s,
        filepath,
        filepath_out,
        fsample,
        t1,
        t2,
        t2_low,
        threads,
        pp_output,
        composite = False,
        argweaver = True,
    ):
        """Run GhostHMM on target and outgroup populations and record results by .

        sample_inds: if None, use all individuals in the tree to run ECM. Otherwise use sample_inds + target_ind
        to run ECM and decode only target_ind.
        """
        if argweaver:
            target_ts = tszip.decompress(str(filepath) + ".0.tsz")
            outgroup_ts = tszip.decompress(str(filepath_out) + ".0.tsz")
        else:
            target_ts = tskit.load(str(filepath) + "_0.trees")
            outgroup_ts = tskit.load(str(filepath_out) + "_0.trees")
        a_param = np.zeros(shape=(inds, 10, 181))
        f_param = np.zeros(shape=(inds, 10, 181))
        acps = np.zeros(shape=(inds, 3))
        fcps = np.zeros(shape=(inds, 3))
        a_xss_rec = np.zeros(shape=(inds, int(target_ts.sequence_length / s + (target_ts.sequence_length % s > 0)), target_ts.num_samples - 1))
        f_xss_rec = np.zeros(shape=(inds, int(outgroup_ts.sequence_length / s + (outgroup_ts.sequence_length % s > 0)), outgroup_ts.num_samples - 1))
        n_gamma = np.zeros(shape=(inds, int(target_ts.sequence_length / s + (target_ts.sequence_length % s > 0))))
        f_gamma = np.zeros(shape=(inds, int(outgroup_ts.sequence_length / s + (outgroup_ts.sequence_length % s > 0))))
        for i in range(inds):
            if composite:
                input_inds = range(inds)
            else:
                input_inds = [i]
            (
                gammas,
                a_res_dict,
                a_xss,
                treespan,
                at2_low, 
                acp, 
                aprop
            ) = self.run_hmm_avg_trees(
                inds = input_inds,
                s = s, 
                filepath = filepath,
                fsample = fsample,
                t1 = int(t1),
                t2 = int(t2),
                t2_low=int(t2_low),
                njob = threads,
                argweaver=argweaver,
            )
            (
                fgammas,
                f_res_dict,
                f_xss,
                ftreespan,
                ft2_low, 
                fcp, 
                fprop
            ) = self.run_hmm_avg_trees(
                inds = input_inds,
                s = s, 
                filepath = filepath_out,
                fsample = fsample,
                t1 = int(t1),
                t2 = int(t2),
                t2_low=int(t2_low),
                njob = threads,
                argweaver=argweaver,
            )
            a_param[i] = pd.DataFrame.from_dict(a_res_dict).to_numpy().T
            f_param[i] = pd.DataFrame.from_dict(f_res_dict).to_numpy().T
            acps[i] = np.array([at2_low, acp, aprop])
            fcps[i] = np.array([ft2_low, fcp, fprop])
            if composite:
                n_gamma = gammas[:, 1, :]
                f_gamma = fgammas[:, 1, :]
                a_xss_rec = a_xss[input_inds]
                f_xss_rec = f_xss[input_inds]
                break
            else:
                n_gamma[i] = gammas[0, 1, :]
                f_gamma[i] = fgammas[0, 1, :]
                a_xss_rec[i] = a_xss
                f_xss_rec[i] = f_xss
        np.savez_compressed(
            str(pp_output),
            a_gamma=n_gamma,
            f_gamma=f_gamma,
            a_xss=a_xss_rec,
            f_xss=f_xss_rec,
            a_param=a_param,
            f_param=f_param,
            a_tree_span=treespan,
            f_tree_span=ftreespan,
            t_admix=t1,
            t_archaic=t2,
            acp=acp,
            fcp=fcp,
            allow_pickle=True
        )
        return n_gamma, treespan, f_gamma, ftreespan