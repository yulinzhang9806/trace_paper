"""Utils for running simulation-based analysis workflow."""
import json
import numpy as np
import pandas as pd
import tskit
import tszip
from tqdm import tqdm
from pathlib import Path
import sys
import os
import pybedtools
from pyfaidx import Fasta
from .trace import TRACE

class Performance_utils:
    """Utilities for performance inference from intervals."""

    def __init__(
        self,
        pp=None,
        treespan=None,
        nodes=None,
    ):
        """Initialize the performance utils class."""
        self.treespan = treespan
        self.pp = pp
        self.nodes = nodes

    def calculate_performance(self, i, t):
        """Calculate recall and precision from tuple objects that are passed in.

        The inferred and truth lists are lists of tuples.
        """
        inferred = IntervalTree.from_tuples(i)
        inferred.merge_overlaps()
        truth = IntervalTree.from_tuples(t)
        truth.merge_overlaps()
        total_HMM = sum([x[1] - x[0] for x in (inferred)])
        truth_seq = sum([x[1] - x[0] for x in (truth)])
        true_positives = []
        false_discovery = []
        for x in inferred:
            overlap = truth.overlap(x)
            if len(overlap) > 0:
                for seg in list(overlap):
                    true_positives.append(min([seg[1], x[1]]) - max([seg[0], x[0]]))
            else:
                false_discovery.append(x[1] - x[0])
        if total_HMM == 0:
            precision = np.nan
        else:
            precision = sum(true_positives) / float(total_HMM)
        if truth_seq == 0:
            recall = np.nan
        else:
            recall = sum(true_positives) / float(truth_seq)
        return (
            precision,
            recall,
            true_positives,
            false_discovery,
            truth_seq,
            total_HMM,
        )

    def read_truth_bed(self, file, popsize):
        """Read truth from bed file."""
        infile = open(file)
        lines = infile.readlines()
        infile.close()
        truth = dict({i: [] for i in range(popsize)})
        popmerge = False
        if len(lines) > 0:
            for i in range(len(lines)):
                if lines[i].startswith("chromosome") or lines[i].startswith("chrom"):
                    continue
                s = lines[i].strip("\n").strip("\t").split("\t")
                if len(s) == 4:
                    if not float(s[1]) == float(s[2]) and int(s[3]) in range(popsize):
                        truth[int(s[3])].append([float(s[1]), float(s[2])])
                elif len(s) > 4:
                    if not float(s[1]) == float(s[2]) and int(s[3]) in range(popsize) and str(s[4]) == 'Ghost':
                        truth[int(s[3])].append([float(s[1]), float(s[2])])
                else:
                    popmerge = True
                    if not float(s[1]) == float(s[2]) and int(s[3]) in range(popsize):
                        truth[0].append([float(s[1]), float(s[2])])
        for i in range(popsize):
            truth[i] = np.array(truth[i])
        if popmerge:
            truth = truth[0]
        return truth

    def read_ind_nodes(self, nodes, idx, popsize, total_tree=None):
        """Read numpy array of node id for potential introgressing nodes (the first node < t_admix for the focal lineage) of one individual.

        nodes: 1-d array of node ids for one individual.

        return: numpy array for node id, each row should be one individual, each column should be node id for one marginal tree.
        """
        if self.nodes is None:
            if total_tree is None:
                self.nodes = np.zeros(shape=(popsize, len(self.treespan)))
            else:
                self.nodes = np.zeros(shape=(popsize, total_tree))
        self.nodes[idx] = nodes
        return self.nodes

    def maxlen(self, states, pp_cutoff, method="posterior", pp=None, nodes=None):
        """Apply maximum length to recover short segments based on introgression node / posterior prob information.

        method:
        'posterior' -- decide the treespan / block to be introgressed for the focal individual if any individual detects introgression at the region and the local pp > pp_cutoff.
        'nodes' -- decide the treespan / block to be introgressed for the focal individual if any individual detects introgression at the region and it has the same introgressing node.
        states: numpy array of 0/1 indicating states for different individuals at different treespans.

        return numpy array of 0/1.
        """
        if pp is None:
            pp = self.pp
        if nodes is None:
            nodes = self.nodes
        maxlen_out = states.copy()
        if method == "nodes":
            for k in range(states.shape[0]):
                for j in range(states.shape[1]):
                    if states[k][j] == 1:
                        n = nodes[k][j]
                        for i in range(nodes.shape[0]):
                            if nodes[i][j] == n:
                                maxlen_out[i][j] = 1
        if method == "posterior":
            for k in range(states.shape[0]):
                for j in range(states.shape[1]):
                    if states[k][j] == 1:
                        for i in range(pp.shape[0]):
                            if pp[i][j] >= pp_cutoff:
                                maxlen_out[i][j] = 1
        self.maxlen_out = maxlen_out
        return maxlen_out

    def get_filtered_tracts(self, states, treespan):
        """Read states matrix and get filtered tracts based on treespan."""
        out = []
        states = (states > 0).astype(int)
        for i in range(states.shape[0]):
            s = states[i]
            j = 0
            ind_out = []
            while j < len(s):
                if s[j] == 1:
                    t = j
                    temp_pos = []
                    while t < len(s) and s[t] == 1:
                        temp_pos.append(t)
                        t += 1
                    ind_out.append(
                        (
                            treespan[np.min(temp_pos)][0],
                            treespan[np.max(temp_pos)][1],
                        )
                    )
                    j = t
                else:
                    j += 1
            out.append(ind_out)
        return out

    def filter_hmm_output(
        self,
        arc_cutoff=0.5,
        pp_cutoff=0.9,
        l_cutoff=0.03,
        popmerge=False,
        maxlen=None,
        combined_pp=None,
        treespan=None,
    ):
        """Filter HMM output based on pp cutoff and length cutoff for each chromosome, all individuals.

        combined_pp: numpy array for posterior probabilities, each row should be an
        individual, each column should be posterior probability for a tree.

        treespan: a numpy array, each row should be a tree, the first column should be
        tree.left.interval, the second column should be tree.right.interval. Trees should be
        ordered in increasing order (or default order returned by ts.trees()).

        maxlen:
        None -- do not apply maxlen filters
        'posterior' -- see maxlen
        'nodes' -- see maxlen

        popmerge: if True, would return a list of tuples merging all archaic regions in the
        population; if False, would return a list of lists of tuples, each list represent
        archaic regions in an individual.
        """
        out = []
        states = np.zeros(shape=(combined_pp.shape[0], combined_pp.shape[1]))
        # apply pp_cutoff and l_cutoff to get states
        ns = 1
        for k in range(combined_pp.shape[0]):
            pp = combined_pp[k]
            i = 0
            while i < combined_pp.shape[1]:
                if pp[i] >= pp_cutoff:
                    j = i
                    temp_pos = []
                    temp_pp = []
                    while (
                        j < len(pp) and pp[j] >= arc_cutoff
                    ):
                        temp_pos.append(j)
                        temp_pp.append(pp[j])
                        j += 1
                    if (
                        np.mean(temp_pp) >= pp_cutoff 
                        and treespan[np.max(temp_pos)][1] 
                        - treespan[np.min(temp_pos)][0]
                        >= l_cutoff
                    ):
                        for t in temp_pos:
                            states[k][t] = ns
                    i = j
                    ns += 1
                else:
                    i += 1
        # apply maxlen filters
        if maxlen is None:
            out = self.get_filtered_tracts(states, treespan)
        else:
            maxlen_out = self.maxlen(
                method=maxlen,
                states=states,
                pp_cutoff=pp_cutoff,
                pp=combined_pp,
                nodes=None,
            )
            out = self.get_filtered_tracts(maxlen_out, treespan)
        # return output
        if popmerge:
            output = out[0]
            for i in range(1, len(out)):
                output = output + out[i]
            tree = IntervalTree.from_tuples(output)
            tree.merge_overlaps()
            tree = list(sorted(tree))
            output = []
            for i in range(len(tree)):
                output.append((tree[i][0], tree[i][1]))
            return output
        else:
            return out, states

class ARG_utils:
    """Class defining functions for tree information extraction."""

    def __init__(
        self,
        total_sample_size=200,  # haplotype number, not individual number
        afr_size=200,
        eur_size=0,
        afr_poplabel=0,
        eur_poplabel=1,
        nea_poplabel=2,
        ghost_poplabel=4,
        human_ancestor_poplabel=1,
    ):
        """Initialize the ARG_funcs class."""
        self.afr_size = afr_size
        self.eur_size = eur_size
        self.afr_poplabel = afr_poplabel
        self.eur_poplabel = eur_poplabel
        self.nea_poplabel = nea_poplabel
        self.human_ancestor_poplabel = human_ancestor_poplabel
        self.ghost_poplabel = ghost_poplabel
        self.total_sample_size = total_sample_size
        self.ts = None
        self.t = None

    def add_tree_sequence(self, ts):
        """Add in a tree-sequence for analysis."""
        self.ts = ts
        self.afr_samples = ts.samples(self.afr_poplabel)
        self.eur_samples = ts.samples(self.eur_poplabel)
        assert ts.num_trees > 1
        self.m = ts.num_trees
        self.pos = np.zeros(self.m)
        self.treespan = dict()
        for i, t in enumerate(ts.trees()):
            self.pos[i] = (t.interval.right + t.interval.left) / 2.0
            self.treespan[i] = np.array([t.interval.left, t.interval.right])
        self.ne_seg = {}
        for i in range(self.total_sample_size):
            self.ne_seg[i] = []
        self.find_intro_trees()

    def find_intro_trees(self):
        """Get introgression tree id and node id."""
        self.tree_afr = set()
        self.tree_common = set()
        self.tree_null = set()
        self.afr_tree_node = dict({})
        self.common_tree_node = dict({})
        for mr in self.ts.migrations():
            if (
                mr.source == self.human_ancestor_poplabel
                and mr.dest == self.ghost_poplabel
            ):
                for tree in self.ts.trees(leaf_lists=True):
                    if mr.left > tree.interval.right:
                        continue
                    if mr.right <= tree.interval.left:
                        break
                    if tree.index in self.tree_common:
                        self.common_tree_node[tree.index].append(mr.node)
                    else:
                        self.tree_common.add(tree.index)
                        self.common_tree_node[tree.index] = [mr.node]
            if mr.source == self.afr_poplabel and mr.dest == self.ghost_poplabel:
                for tree in self.ts.trees(leaf_lists=True):
                    if mr.left > tree.interval.right:
                        continue
                    if mr.right <= tree.interval.left:
                        break
                    if tree.index in self.tree_afr:
                        self.afr_tree_node[tree.index].append(mr.node)
                    else:
                        self.tree_afr.add(tree.index)
                        self.afr_tree_node[tree.index] = [mr.node]
        for tree in self.ts.trees():
            if tree.index not in self.tree_common and tree.index not in self.tree_afr:
                self.tree_null.add(tree.index)

    def extract_tmrca(self, ingroup, outgroup, tree):
        """Extract TMRCA for one individual."""
        assert self.ts is not None
        mean_tmrca = []
        var_tmrca = []
        med_tmrca = []
        all_tmrca = []
        for inind in ingroup:
            #            all_tmrca = []
            for outind in outgroup:
                if inind != outind:
                    t = tree.tmrca(inind, outind)
                    all_tmrca.append(t)
            mean_tmrca.append(np.mean(all_tmrca))
            var_tmrca.append(np.var(all_tmrca))
            med_tmrca.append(np.median(all_tmrca))
        assert len(mean_tmrca) == len(var_tmrca) == len(med_tmrca) == len(ingroup)
        return mean_tmrca, var_tmrca, med_tmrca, all_tmrca

    def extract_tmrca_all(self):
        """Extract TMRCA for all individuals."""
        out_afr = [[], [], [], []]
        out_common = [[], [], [], []]
        out_null = [[], [], [], []]
        out = [out_common, out_afr, out_null]
        for tree in self.ts.trees():
            afr = []
            i = 0
            if tree.index in self.tree_common:
                i = 0
                for node in self.common_tree_node[tree.index]:
                    for l in tree.samples(node):
                        afr.append(l)
            elif tree.index in self.tree_afr:
                i = 1
                for node in self.afr_tree_node[tree.index]:
                    for l in tree.samples(node):
                        afr.append(l)
            else:
                i = 2
                afr = self.afr_samples[0:2]
            m, v, med, tr = self.extract_tmrca(afr, self.afr_samples, tree)
            out[i][0] = out[i][0] + m
            out[i][1] = out[i][1] + v
            out[i][2] = out[i][2] + med
            #            if len(tr) > 0 and len(out[i][3]) == 0:
            #                out[i][3] = tr
            out[i][3] = out[i][3] + tr
        return out[0], out[1], out[2]

    def extract_coalescent_counts(self, target, pop, t_admix, t_archaic, tree):
        """Extract number of coalescent events in the time interval defined by the two Ts for one individual."""
        assert self.ts is not None
        tmrcas = []
        u = target
        while u != tskit.NULL:
            tmrcas.append(tree.time(u))
            u = tree.parent(u)
        tmrcas = np.array(tmrcas)
        if max(tmrcas) <= t_admix:
            c = 7
        else:
            c = ((t_admix < tmrcas) & (tmrcas < t_archaic)).sum()
        return c

    def extract_coalescent_counts_ind(self, target, pop, t_admix, t_archaic, tree):
        """Extract number of coalescent individuals in the time interval defined by the two Ts for one individual."""
        assert self.ts is not None
        tmrcas = []
        nids = []
        c = 0
        u = target
        while u != tskit.NULL:
            tmrcas.append(tree.time(u))
            nids.append(u)
            u = tree.parent(u)
        tmrcas = np.array(tmrcas)
        nids = np.array(nids)
        n1 = nids[np.where(tmrcas <= t_admix)]
        n2 = nids[np.where(tmrcas <= t_archaic)]
        c = tree.num_samples(n2[-1]) - tree.num_samples(n1[-1])
        # if tmrcas[-1] >= t_admix:
        #     n = nids[np.where((tmrcas >= t_admix) & (tmrcas <= t_archaic))]
        #     if len(n) > 0:
        #         c = tree.num_samples(n[-1])
        # else:
        #     c = np.nan
        return c

    def extract_coalescent_counts_all(self, t_admix, t_archaic):
        """Extrat number of coalescent events in the time interval defined by the two Ts for all individuals and trees."""
        out_afr = []
        out_common = []
        out_null = []
        out = [out_common, out_afr, out_null]
        for tree in self.ts.trees():
            afr = []
            i = 0
            if tree.index in self.tree_common:
                i = 0
                for node in self.common_tree_node[tree.index]:
                    for l in tree.samples(node):
                        afr.append(l)
            elif tree.index in self.tree_afr:
                i = 1
                for node in self.afr_tree_node[tree.index]:
                    for l in tree.samples(node):
                        afr.append(l)
            else:
                i = 2
                afr = self.afr_samples[0:30]
            for j in afr:
                # f = self.extract_coalescent_counts_ind(
                #         j, self.afr_samples, t_admix, t_archaic, tree
                #     )
                # if not np.isnan(f):
                #     out[i].append(f)
                out[i].append(
                    self.extract_coalescent_counts(
                        j, self.afr_samples, t_admix, t_archaic, tree
                    )
                )

        return out[0], out[1], out[2]

    def get_longest_branch(self, afrid, cond, tree):
        """Get longest branch length from one individual where the lower end is less than some condition."""
        longest = []
        longest_lower = []
        longest_upper = []
        for afr in afrid:
            branches = {}
            ll = afr
            while tree.time(ll) < cond:
                branches[tree.time(ll)] = [
                    tree.time(ll),
                    tree.time(tree.parent(ll)),
                    tree.branch_length(ll),
                ]
                ll = tree.parent(ll)
            longest.append(max(branches.keys()))
            longest_lower.append(branches[max(branches.keys())][0])
            longest_upper.append(branches[max(branches.keys())][1])
        return longest, longest_lower, longest_upper

    def extract_branch_length_all(self, condition):
        """Extract branch length from all individuals."""
        out_afr = [[], [], []]
        out_common = [[], [], []]
        out_null = [[], [], []]
        out = [out_common, out_afr, out_null]
        for tree in self.ts.trees():
            afr = []
            i = 0
            if tree.index in self.tree_common:
                i = 0
                for node in self.common_tree_node[tree.index]:
                    for l in tree.samples(node):
                        afr.append(l)
            elif tree.index in self.tree_afr:
                i = 1
                for node in self.afr_tree_node[tree.index]:
                    for l in tree.samples(node):
                        afr.append(l)
            else:
                i = 2
                afr = self.afr_samples[0:2]
            b, l, u = self.getLongestBranch(afr, condition, tree)
            out[i][0] = out[i][0] + b
            out[i][1] = out[i][1] + l
            out[i][2] = out[i][2] + u
        return out[0], out[1], out[2]

    def extract_branch_boundaries(self, i, cond):
        """Extract the branch length subtending sample i."""
        assert self.ts is not None
        lower_intro = []
        upper_intro = []
        lower_null = []
        upper_null = []
        for tree in self.ts.trees():
            lower = []
            parents = []
            u = i
            while u != tskit.NULL:
                parents.append(u)
                lower.append(tree.time(u))
                u = tree.parent(u)
            if u == tskit.NULL:
                lower.append(tree.time(tree.roots[0]))
                parents.append(tree.roots[0])
            lower = np.array(lower)
            if lower[-1] > cond:
                lower_node = lower[np.argwhere(lower <= cond)[-1]]
                upper_node = lower[np.argwhere(lower >= cond)[0]]
            else:
                lower_node = 0
                upper_node = 0
            if tree.index in self.tree_afr and (
                set(self.afr_tree_node[tree.index]) & set(parents)
            ):
                lower_intro.append(lower_node[0])
                upper_intro.append(upper_node[0])
            else:
                lower_null.append(lower_node[0])
                upper_null.append(upper_node[0])
        assert np.all(lower_null < upper_null)
        assert np.all(np.array(lower_null) <= cond)
        return lower_intro, upper_intro, lower_null, upper_null

    def extract_branch_boundaries_all(self, cond):
        """Extract all conditional branches."""
        lower_intro_all = []
        upper_intro_all = []
        lower_null_all = []
        upper_null_all = []
        for i in self.afr_samples[0:100]:
            (
                lower_intro,
                upper_intro,
                lower_null,
                upper_null,
            ) = self.extract_branch_boundaries(i, cond)
            lower_intro_all = lower_intro_all + lower_intro
            upper_intro_all = upper_intro_all + upper_intro
            lower_null_all = lower_null_all + lower_null
            upper_null_all = upper_null_all + upper_null
        return lower_intro_all, upper_intro_all, lower_null_all, upper_null_all

    def combine_segs(self, j, get_segs=True):
        """Combine introgressed segments."""
        assert self.ts is not None
        segs = np.array(self.ne_seg[j])
        merged = np.empty([0, 2])
        if len(segs) == 0:
            if get_segs:
                return []
            else:
                return 0
        sorted_segs = segs[np.argsort(segs[:, 0]), :]
        for higher in sorted_segs:
            if len(merged) == 0:
                merged = np.vstack([merged, higher])
            else:
                lower = merged[-1, :]
                if higher[0] <= lower[1]:
                    upper_bound = max(lower[1], higher[1])
                    merged[-1, :] = (lower[0], upper_bound)
                else:
                    merged = np.vstack([merged, higher])
        if get_segs:
            self.ne_seg[j] = merged
            return merged
        else:
            return np.sum(merged[:, 1] - merged[:, 0]) / self.ts.sequence_length

    def write_bed_output(self, name, segs, chrom=1, indinfo=False, cm=False):
        """Write result to bed file."""
        if indinfo:
            with open(name, "w+") as out:
                for ind in segs.keys():
                    if not len(segs[ind]) == 0:
                        for se in segs[ind]:
                            if cm:
                                out.write(
                                    "\t".join(
                                        [str(chrom)]
                                        + [str(int(round(j)) * 1e-6) for j in se]
                                        + [str(ind)]
                                    )
                                    + "\n"
                                )
                            else:
                                out.write(
                                    "\t".join(
                                        [str(chrom)]
                                        + [str(int(round(j))) for j in se]
                                        + [str(ind)]
                                    )
                                    + "\n"
                                )
        else:
            bstring = ""
            for ind in segs.keys():
                if not len(segs[ind]) == 0:
                    bstring += self.write_bed_string(segs[ind], cm=cm)
            with open(name, "w+") as out:
                out.write(bstring)

    def write_bed_string(self, segs, chrom=1, cm=False):
        """Write a bed-style string (avoiding file output)."""
        outstring = ""
        for se in segs:
            if cm:
                outstring += (
                    "\t".join([str(chrom)] + [str(int(round(j)) * 1e-6) for j in se])
                    + "\n"
                )
            else:
                outstring += (
                    "\t".join([str(chrom)] + [str(int(round(j))) for j in se]) + "\n"
                )
        return outstring

    def extract_ghost_intro_all(self, from_pop, to_pop):
        """Extract ghost introgression based on migration events."""
        assert self.ts is not None
        for mr in self.ts.migrations():
            if mr.source == from_pop and mr.dest == to_pop:
                for tree in self.ts.trees(leaf_lists=True):
                    if mr.left > tree.interval.right:
                        continue
                    if mr.right <= tree.interval.left:
                        break
                    for i in tree.samples(mr.node):
                        left = max([tree.interval.left, mr.left])
                        right = min([tree.interval.right, mr.right])
                        self.ne_seg[i].append([left, right])
        for i in self.ne_seg:
            true_ne_segs = self.combine_segs(j=i)
            self.ne_seg[i] = true_ne_segs

    def get_ghost_intro_ind(self, i, all_intro=None):
        """Extract ghost introgression for individual."""
        if all_intro is None:
            all_intro = self.ne_seg
        if not len(all_intro) > 0:
            print('Please run "extract_ghost_intro_all" first.')
        return all_intro[i]

    def filter_pp_tmrca(
        self, pos, tmrca_pp, pp_cutoff=0.9, l_cutoff=5e4, w_s=0.5, w_l=1
    ):
        """Filter tmrca_pp with haplotype length."""
        i = 0
        assert len(pos) == len(tmrca_pp)
        out_weight = []
        while i < len(tmrca_pp):
            if tmrca_pp[i] >= 0.5:
                j = i
                temp_pos = []
                temp_pp = []
                while j < len(tmrca_pp) and tmrca_pp[j] >= 0.5:
                    temp_pos.append(pos[j])
                    temp_pp.append(tmrca_pp[j])
                    j += 1
                if (
                    np.mean(temp_pp) >= pp_cutoff
                    and np.max(temp_pos) - np.min(temp_pos) >= l_cutoff
                ):
                    for t in range(i, j):
                        out_weight.append(w_l)
                else:
                    for t in range(i, j):
                        out_weight.append(w_s)
                i = j
            else:
                out_weight.append(w_s)
                i += 1
        return out_weight

    def combine_pp(self, tmrca_pp, branch_pp, tmrca_weight, w_s=0.0, w_l=0.7):
        """Combine posterior probability of two hmm."""
        assert len(tmrca_pp) == len(branch_pp)
        assert len(tmrca_weight) == len(tmrca_pp)
        out_pp = []
        for b, t, w in zip(branch_pp, tmrca_pp, tmrca_weight):
            if b > 0.5:
                out_pp.append(w_l * b + (1 - w_l) * t * w)
            else:
                out_pp.append(w_s * b + (1 - w_s) * t * w)
        return np.array(out_pp)

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