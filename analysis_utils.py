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

    def extract_tmrca(input_tsz, inds, other_samples, outpref, windowsize=1000):
        def get_pairwise_times(ts, windowsize, k1, k2):
            """Function for getting pairwise average tmrca over the genome."""
            windows = np.arange(0, ts.sequence_length, windowsize)
            windows = np.append(windows, ts.sequence_length)
            times = ts.divergence(sample_sets=[k1, k2], windows=windows, mode='branch')
            return windows, 0.5*times
        if input_tsz.endswith(".tsz"):
            ts = tszip.decompress(input_tsz)
        else:
            ts = tskit.load(input_tsz)
        out = np.zeros(shape=(len(inds), len(other_samples), int(ts.sequence_length / windowsize) + (1 if ts.sequence_length % windowsize > 0 else 0)))
        for i in range(len(inds)):
            k1 = inds[i]
            os = [s for s in other_samples if s != k1]
            for j in range(len(os)):
                k2 = os[j]
                windows, times = get_pairwise_times(ts, windowsize, [k1], [k2])
                out[i, j, :] = times
        np.savez_compressed(f"{outpref}.npz", tmrca=out, windows=windows, inds=inds, other_samples=other_samples)

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

class ExplicitDefaultsHelpFormatter(argparse.ArgumentDefaultsHelpFormatter):
    """
    format the help menu such that defaults are printed to help
    only when they are explicitly stated in the parser
    (e.g. do not print actions or Nones)
    """

    def _get_help_string(self, action):
        """
        returns the help string
        """

        if action.default in (None, False):
            return action.help
        return super()._get_help_string(action)

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
