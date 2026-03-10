"""Utility functions for SNP info annotation."""
import os
import numpy as np

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
        os.systme(
            "bcftools query -l " + str(archaicbcf) + ".bcf" + " > " + snpinfo + "archaic_samples"
        )
        with open(snpinfo + "archaic_samples") as f:
            archaic_samples = f.read().strip().split('\n')
        arc_return_dict = {x: idx for idx, x in enumerate(archaic_samples)}
        os.remove(snpinfo + "archaic_samples")
        existing_sites, geno_info = self.parse_archaic_geno(snpinfo + "archaic_geno", arc_return_dict)
        infile=open(snpinfo + '.txt')
        lines=infile.readlines()
        infile.close()
        out = lines[0].strip('\n') + '\t' + '\t'.join(archaic_samples) + '\n'
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