"""Utility functions for TRACE benchmarking and data processing."""
import os
import numpy as np
import pandas as pd
import tskit
from intervaltree import Interval, IntervalTree
from numpy import mean, median, var
import sys
import argparse
import json
from tqdm import tqdm
from pathlib import Path
from trace.trace import TRACE
import pybedtools


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

class Figure_utils:
    """Utility for data processing and figure generation."""

    def bootstrap(self, data, nboot, ntime, func=np.mean, replace=True):
        """Bootstrapping function. Returns a 1-dimensional numpy array with ntime entries.

        data: 1-dimensional numpy array including all data points.
        nboot: bootstrap sample size.
        ntime: times repeating the bootstrap process.
        func: function applied to the bootstrapping samples.
        replace: sample with replacement or not.
        """
        out = []
        for i in range(ntime):
            sample = np.random.choice(data, nboot, replace=replace)
            out.append(func(sample))
        return np.array(out)

    def merge_seeds(self, filepath, t_admix, t_archaic, seeds):
        """Merge datafiles from simulations of different seeds."""
        out_df = pd.read_csv(
            str(filepath)
            + str(seeds[0])
            + "_"
            + str(t_admix)
            + "_"
            + str(t_archaic)
            + "_data.tsv",
            sep="\t",
        )
        out_df["seed"] = seeds[0]
        for s in range(1, len(seeds)):
            df = pd.read_csv(
                str(filepath)
                + str(seeds[s])
                + "_"
                + str(t_admix)
                + "_"
                + str(t_archaic)
                + "_data.tsv",
                sep="\t",
            )
            df["seed"] = seeds[s]
            out_df = pd.concat([out_df, df])
        return out_df

    def cut_dataframe(self, df, pp_cutoff, l_cutoff):
        """Cut dataframe according to posterior probability threshold and haplotype length threshold. Remove na."""
        out_df = df[
            (df["mean_posterior_cutoff"] == pp_cutoff)
            & (df["length_cutoff"] == l_cutoff)
        ]
        out_df = out_df.dropna(subset=["recall"])
        return out_df

    def get_datapoints(self, df, colname, ntime, pp_cutoff, l_cutoff):
        """Get datapoints for different length cutoffs with stderr for plotting."""
        out = []
        outerr = []
        for ll in l_cutoff:
            subdf = self.cut_dataframe(df=df, pp_cutoff=pp_cutoff, l_cutoff=ll)
            pres = self.bootstrap(
                data=subdf[colname].to_numpy(),
                nboot=len(subdf),
                ntime=ntime,
                func=np.nanmean,
                replace=True,
            )
            out.append(np.nanmean(pres))
            outerr.append(
                np.sqrt((np.std(pres) ** 2) / (len(pres) - np.sum(np.isnan(pres))))
            )
        return np.array(out), np.array(outerr)

    def remove_spines(self, ax, right=True, top=True):
        """Remove right and/or upper spines of subplots."""
        if right:
            ax.spines["right"].set_visible(False)
        if top:
            ax.spines["top"].set_visible(False)

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
        treespan_cM = TRACE().add_recombination_map(treespan_phy, recombrate)
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