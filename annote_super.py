import numpy as np
import sys
import pandas as pd
import pybedtools
import os

class SUMMARIZE:
    """A class of functions to summarize the results."""

    def filter_tracts(
        self,
        indiv_pp,
        treespan,
        treespan_phy,
        pp_cutoff=0.9,
        arc_cutoff=0.5,
        phy_cutoff=5e4,
        l_cutoff=0.05,
        remove_margin=0,
    ):
        tracts = []
        states = np.zeros(indiv_pp.shape[0])
        i = 0
        while i < indiv_pp.shape[0]:
            if indiv_pp[i] >= pp_cutoff:
                j = i
                temp_pos = []
                temp_pp = []
                while (
                    j < len(indiv_pp) and indiv_pp[j] >= arc_cutoff
                ):
                    temp_pos.append(j)
                    temp_pp.append(indiv_pp[j])
                    j += 1
                if (
                    np.mean(temp_pp) >= pp_cutoff 
                    and treespan[np.max(temp_pos)][1] 
                    - treespan[np.min(temp_pos)][0]
                    >= l_cutoff
                    and treespan_phy[np.max(temp_pos)][1]
                    - treespan_phy[np.min(temp_pos)][0]
                    >= phy_cutoff
                    ):
                    start = treespan_phy[temp_pos[0]][0]
                    end = treespan_phy[temp_pos[-1]][1]
                    tracts.append([start, end, np.mean(temp_pp), end - start, treespan[temp_pos[-1]][1] - treespan[temp_pos[0]][0]])
                    states[np.array(temp_pos[remove_margin:(len(temp_pos) - remove_margin)])] = 1
                i = j
            else:
                i += 1
        return tracts, states

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
            hmmix = hmmix[hmmix["ID"] == individualID]
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
            nea_missing = (s[arc_return_dict['Chagyrskaya-Phalanx']] == '9') & (s[arc_return_dict['AltaiNeandertal']] == '9') & (s[arc_return_dict['Vindija33.19']] == '9')
            dden = (s[arc_return_dict['Denisova']] in ['1','2'])
            den_missing = (s[arc_return_dict['Denisova']] == '9')
            aafr = float(s[-3])
            aout = False if float(s[-2]) >= 0.05 else True
            instrict = True if int(s[-1]) == 1 else False
            freqs = np.array([float(s[15]), float(s[16]), float(s[17])])
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
            if str(s[1]) == "N" or str(s[2]) == "N":
                continue # skip weird SNPs with N alleles
            if not pos in snp_dict:
                print(pos)
                print("Error: SNP not found in snp_dict.")
                sys.exit(1)
            if str(s[3]) in ['./.', './1', './0', '0/.', '1/.']: # skip missing sites
                continue
            delimiter = "/" if "/" in str(s[3]) else "|"
            genos = str(s[3]).split(delimiter)
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
        if len(lines) >= 2:
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

    def append_seg_freq(self, npzpref, chrom, indlist, summary, colname, outpref):
        """Append segment frequency in defined population to the summary file."""
        with np.load(f"{npzpref}{indlist[0]}.{chrom}.xss.npz") as data:
            treespan_phy = data["treespan_phy"]
        windowsize = treespan_phy[0][1] - treespan_phy[0][0]
        states = np.zeros((len(indlist), treespan_phy.shape[0]))
        for idx in range(len(indlist)):
            with np.load(f"{npzpref}{indlist[idx]}.{chrom}.xss.npz") as data:
                try:
                    s = data['states']
                    states[idx] = s
                except:
                    print(f"Error: {indlist[idx]} does not have states in {npzpref}{indlist[idx]}.{chrom}.xss.npz")
                    continue
        sum_states = np.sum(states, axis=0)
        freq_states = np.mean(states, axis=0)
        out = "chromosome\tstart\tend\tcount\tfreq\n"
        for i in range(treespan_phy.shape[0]):
            out += f"{chrom}\t{treespan_phy[i][0]}\t{treespan_phy[i][1]}\t{sum_states[i]}\t{freq_states[i]}\n"
        outfile=open(outpref + '.txt','w')
        outfile.write(out)
        outfile.close()
        infile=open(summary + ".txt")
        lines=infile.readlines()
        infile.close()
        out = lines[0].strip('\n') + "\t" + str(colname) + "\n"
        for i in range(1, len(lines)):
            s = lines[i].strip('\n').strip('\t').split('\t')
            st = int(int(s[1]) / windowsize)
            ed = int(int(s[2]) / windowsize)
            me = np.mean(freq_states[st:ed])
            out += lines[i].strip('\n') + "\t" + str(me) + '\n'
        outfile=open(summary + "_" + colname + '.txt','w')
        outfile.write(out)
        outfile.close()

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

    def merge_df_count_branch_mut(self, count, count_t1t2, output, pop="AFR", unrec_label = "Ghost"):
        data = pd.read_csv(count_t1t2, sep="\s+")
        dff = pd.read_csv(count, sep="\s+")
        if pop == "AFR":
            data = data.merge(dff, on=dff.columns[0:11].tolist(), how='left')
        elif pop == "EUR" or pop == "OCN" or pop == "EAS" or pop == "SEA":
            data = data.merge(dff, on=dff.columns[0:16].tolist(), how='left')
        else:
            raise ValueError("Unsupported population type. Please use AFR, EUR, OCN, EAS, or SEA.")
        nd00_aff = []
        nd10_aff = []
        nd01_aff = []
        nd11_aff = []
        yri_aff = []
        tot_aff = []
        for i in data.index:
            nd00 = 0
            nd10 = 0
            nd01 = 0
            nd11 = 0
            nyri = 0
            tot = 0
            mks = str(data.loc[i, 'dsnps_marks']).split(',')
            ons = str(data.loc[i, 'branch_mark']).split(',')
            yri = str(data.loc[i, 'DAF_YRI']).split(',')
            for j in range(len(mks)):
                if ons[j] == 'on' and mks[j] in ['ND00_strict', 'ND10_strict', 'ND01_strict', 'ND11_strict']:
                    tot += 1
                    if mks[j] == 'ND00_strict':
                        nd00 += 1
                    elif mks[j] == 'ND10_strict':
                        nd10 += 1
                    elif mks[j] == 'ND01_strict':
                        nd01 += 1
                    elif mks[j] == 'ND11_strict':
                        nd11 += 1
                    if float(yri[j]) > 0:
                        nyri += 1
            if tot == 0:
                nd00_aff.append(np.nan)
                nd10_aff.append(np.nan)
                nd01_aff.append(np.nan)
                nd11_aff.append(np.nan)
                tot_aff.append(np.nan)
                yri_aff.append(np.nan)
            else:
                nd00_aff.append(nd00)
                nd10_aff.append(nd10)
                nd01_aff.append(nd01)
                nd11_aff.append(nd11)
                tot_aff.append(tot)
                yri_aff.append(nyri)
        nd00_aff = np.array(nd00_aff)
        nd10_aff = np.array(nd10_aff)
        nd01_aff = np.array(nd01_aff)
        nd11_aff = np.array(nd11_aff)
        tot_aff = np.array(tot_aff)
        yri_aff = np.array(yri_aff)
        data['nd00_b'] = nd00_aff
        data['nd10_b'] = nd10_aff
        data['nd01_b'] = nd01_aff
        data['nd11_b'] = nd11_aff
        data['tot_b'] = tot_aff
        data['yri_b'] = yri_aff
        data['nea_prop'] = (nd10_aff + nd11_aff) / tot_aff
        data['den_prop'] = (nd01_aff + nd11_aff) / tot_aff
        data['nd00_prop'] = nd00_aff / tot_aff
        data['nd10_prop'] = nd10_aff / (nd00_aff + nd10_aff + nd01_aff)
        data['nd01_prop'] = nd01_aff / (nd00_aff + nd10_aff + nd01_aff)
        data['nYRI'] = yri_aff / tot_aff
        data['assign_label'] = "NONE"
        data['nea_aff'] = (data['ND10_strict'] + data['ND11_strict']) / (data['ND00_strict'] + data['ND10_strict'] + data['ND01_strict'] + data['ND11_strict'])
        data['den_aff'] = (data['ND01_strict'] + data['ND11_strict']) / (data['ND00_strict'] + data['ND10_strict'] + data['ND01_strict'] + data['ND11_strict'])
        if unrec_label == "SUPER":
            data.loc[(data['nd10_prop'] > 0.2) & (data['ND10_strict'] > data['ND01_strict']) & (data['nderived_strict'] >= 10) & (data["tot_b"] >= 10) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 0), 'assign_label'] = "NEA"
            data.loc[(data['nd01_prop'] > 0.2) & (data['ND10_strict'] < data['ND01_strict']) & (data['nderived_strict'] >= 10) & (data["tot_b"] >= 10) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 0), 'assign_label'] = "DEN"
            data.loc[(data['nd10_prop'] <= 0.2) & (data['nd01_prop'] <= 0.2) & (data["nea_aff"] < 0.5) & (data["den_aff"] < 0.5) & (data["nYRI"] < 0.2) & (data['nderived_strict'] >= 10) & (data["tot_b"] >= 10) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 0), 'assign_label'] = unrec_label
        elif unrec_label == "Ghost":
            data.loc[((data['nd00_prop'] <= 0.8) | (data['nYRI'] <= 0.1)) & (data['ND10_strict'] > data['ND01_strict']) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = "NEA"
            data.loc[((data['nd00_prop'] <= 0.8) | (data['nYRI'] <= 0.1)) & (data['ND10_strict'] < data['ND01_strict']) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = "DEN"
            data.loc[(data['nd00_prop'] > 0.8) & (data['nYRI'] > 0.1) & (data['nderived_strict'] >= 30) & (data[["ND00_strict", "ND10_strict", "ND01_strict", "ND11_strict"]].sum(axis=1) >= 10), 'assign_label'] = unrec_label
        else:
            raise ValueError("Unsupported unrec_label. Please use SUPER or Ghost.")
        data.to_csv(output, sep="\t", index=False)

ind = int(sys.argv[1])
summarypref = sys.argv[2]
filepref = sys.argv[3]
outpref = sys.argv[4]
chrom = sys.argv[5]
unrec_label = sys.argv[6]

df = pd.read_csv("treeID.txt", sep="\t")
hmmixpath = "/global/secure0/groups/mcb-aux01-access/zhangyulin9806/EVOCEANIA_singer/hmmix/"
inds = [ind]
hmmixfiles = [f"{hmmixpath}{df[df['ID']==inds[i]]['Name'].values[0]}.txt" for i in range(len(inds))]
ibdmixeur = f"/global/secure0/groups/mcb-aux01-access/zhangyulin9806/EVOCEANIA_singer/ibdmix/ibdmix/OCN/{chrom}.txt"
ibdmixafr = f"/global/secure0/groups/mcb-aux01-access/zhangyulin9806/EVOCEANIA_singer/ibdmix/ibdmix/AFR/{chrom}.txt"
xss_15k_pref = f"singerave_t15000_{chrom}_ind"
xss_31k_pref = f"singerave_t31500_{chrom}_ind"
individualID = [df[df['ID']==inds[i]]['SampleID'].values[0] for i in range(len(inds))]

if unrec_label == "SUPER":
    # xssfile = summarypref + xss_31k_pref + str(inds[0]) + f".{chrom}.xss.npz"
    # with np.load(xssfile) as data:
    #     pp = data["gammas"]
    #     treespan = data["treespan"]
    #     treespan_phy = data["treespan_phy"]
    #     window_size = int(treespan_phy[0, 1] - treespan_phy[0, 0])
    # tracts, states = SUMMARIZE().filter_tracts(
    #     indiv_pp=pp,
    #     treespan=treespan,
    #     treespan_phy=treespan_phy,
    #     pp_cutoff=0.9,
    #     arc_cutoff=0.5,
    #     phy_cutoff=1e4,
    #     l_cutoff=0.01,
    #     remove_margin=0,
    # )
    xss_15k = summarypref + xss_15k_pref + str(inds[0]) + f".{chrom}.xss.npz"
    with np.load(xss_15k) as d:
        den_states = d["den_states"]
        nea_states = d["nea_states"]
        treespan = d["treespan"]
        treespan_phy = d["treespan_phy"]
        window_size = int(treespan_phy[0, 1] - treespan_phy[0, 0])
        t2s = d["t2s"]
    t2s = np.median(t2s, axis=0)
    tracts, states = SUMMARIZE().filter_tracts(
        indiv_pp=t2s,
        treespan=treespan,
        treespan_phy=treespan_phy,
        pp_cutoff=31500,
        arc_cutoff=31500,
        phy_cutoff=10000,
        l_cutoff=0.01,
        remove_margin=0,
    )
    den_tracts = []
    nea_tracts = []
    fil_den_states = np.zeros(states.shape)
    fil_nea_states = np.zeros(states.shape)
    for t in tracts:
        start_idx = int(t[0] / window_size)
        end_idx = int(t[1] / window_size)
        # must inside den segs
        if np.all(den_states[start_idx:end_idx] > 0):
            # not on the margin
            if den_states[start_idx - 1] > 0 and den_states[end_idx] > 0:
                # find den tract start and end
                den_start = start_idx
                while den_start > 0 and den_states[den_start - 1] > 0:
                    den_start -= 1
                den_end = end_idx
                while den_end < len(den_states) and den_states[den_end] > 0:
                    den_end += 1
                den_start_phy = treespan_phy[den_start][0]
                den_end_phy = treespan_phy[den_end - 1][1]
                # find prev and pos tracts
                prev_len = t[0] - np.max([den_start_phy, int(t[0] - t[3])])
                pos_len = np.min([den_end_phy, int(t[1] + t[3])]) - t[1]
                # if prev_len >= 5000 and pos_len >= 5000:
                #     den_tracts.append(t)
                #     fil_den_states[start_idx:end_idx] = 1
                den_tracts.append(t)
                fil_den_states[start_idx:end_idx] = 1
        if np.all(nea_states[start_idx:end_idx] > 0):
            # not on the margin
            if nea_states[start_idx - 1] > 0 and nea_states[end_idx] > 0:
                # find nea tract start and end
                nea_start = start_idx
                while nea_start > 0 and nea_states[nea_start - 1] > 0:
                    nea_start -= 1
                nea_end = end_idx
                while nea_end < len(nea_states) and nea_states[nea_end] > 0:
                    nea_end += 1
                nea_start_phy = treespan_phy[nea_start][0]
                nea_end_phy = treespan_phy[nea_end - 1][1]
                # find prev and pos tracts
                prev_len = t[0] - np.max([nea_start_phy, int(t[0] - t[3])])
                pos_len = np.min([nea_end_phy, int(t[1] + t[3])]) - t[1]
                # if prev_len >= 5000 and pos_len >= 5000:
                #     nea_tracts.append(t)
                #     fil_nea_states[start_idx:end_idx] = 1
                nea_tracts.append(t)
                fil_nea_states[start_idx:end_idx] = 1
    out = "chromosome\tstart\tend\tmean_posterior\tlength(bp)\tlength(cM)\n"
    for i in range(len(den_tracts)):
        out += (
            "1\t"
            + str(int(den_tracts[i][0]))
            + "\t"
            + str(int(den_tracts[i][1]))
            + "\t"
            + str(round(den_tracts[i][2], 2))
            + "\t"
            + str(int(den_tracts[i][3]))
            + "\t"
            + str(round(den_tracts[i][4], 3))
            + "\n"
        )
    outfile = open(f"{summarypref}{outpref}{ind}.summary.txt", "w")
    outfile.write(out)
    outfile.close()
    out = "chromosome\tstart\tend\tmean_posterior\tlength(bp)\tlength(cM)\n"
    for i in range(len(nea_tracts)):
        out += (
            "1\t"
            + str(int(nea_tracts[i][0]))
            + "\t"
            + str(int(nea_tracts[i][1]))
            + "\t"
            + str(round(nea_tracts[i][2], 2))
            + "\t"
            + str(int(nea_tracts[i][3]))
            + "\t"
            + str(round(nea_tracts[i][4], 3))
            + "\n"
        )
    outfile = open(f"{summarypref}{outpref}{ind}.nea.summary.txt", "w")
    outfile.write(out)
    outfile.close()

elif unrec_label == "Ghost":
    xssfile = summarypref + xss_15k_pref + str(inds[0]) + f".{chrom}.xss.npz"
    with np.load(xssfile) as d:
        data = {k: d[k] for k in d.files}
    pp = data["gammas"] 
    treespan = data["treespan"]
    treespan_phy = data["treespan_phy"]
    window_size = int(treespan_phy[0, 1] - treespan_phy[0, 0])
    tracts, states = SUMMARIZE().filter_tracts(
        indiv_pp=pp[1],
        treespan=treespan,
        treespan_phy=treespan_phy,
        pp_cutoff=0.9,
        arc_cutoff=0.5,
        phy_cutoff=5e4,
        l_cutoff=0.05,
        remove_margin=0,
    )
    out = "chromosome\tstart\tend\tmean_posterior\tlength(bp)\tlength(cM)\n"
    for i in range(len(tracts)):
        out += (
            str(chrom)
            + "\t"
            + str(int(tracts[i][0]))
            + "\t"
            + str(int(tracts[i][1]))
            + "\t"
            + str(round(tracts[i][2], 2))
            + "\t"
            + str(int(tracts[i][3]))
            + "\t"
            + str(round(tracts[i][4], 3))
            + "\n"
        )
    outfile = open(f"{summarypref}{outpref}{ind}.summary.txt", "w")
    outfile.write(out)
    outfile.close()
    data["states"] = states
    np.savez_compressed(
        f"{summarypref}{xss_15k_pref}{inds[0]}.{chrom}.xss.npz",
        **data,
    )

if unrec_label == "Ghost":
    t = 15000
else:
    t = 31500

for i in range(len(inds)):
    prefix = summarypref + filepref + str(inds[i]) + ".summary"
    if df[df['ID']==inds[i]]['SuperGroup'].values[0] in ["EUR", "OCN", "EAS", "SEA"]:
        prefixn = prefix + ".hmmix"
        SUMMARIZE().append_hmmix_info(hmmixfiles[i], prefix + ".txt", prefixn, inference = "hmmix", individualID = individualID[i])
        prefix = prefixn
        prefixn = prefix + ".ibdmix"
        SUMMARIZE().append_hmmix_info(ibdmixeur, prefix + ".txt", prefixn, inference = "ibdmix", individualID = individualID[i])
        os.remove(prefix + ".txt")
        if unrec_label == "SUPER":
            prefix = prefix = summarypref + filepref + str(inds[i]) + ".nea.summary"
            prefixn = prefix + ".hmmix"
            SUMMARIZE().append_hmmix_info(hmmixfiles[i], prefix + ".txt", prefixn, inference = "hmmix", individualID = individualID[i])
            os.remove(prefix + ".txt")
            prefix = prefixn
            prefixn = prefix + ".ibdmix"
            SUMMARIZE().append_hmmix_info(ibdmixeur, prefix + ".txt", prefixn, inference = "ibdmix", individualID = individualID[i])
    else:
        prefixn = prefix + ".ibdmix"
        SUMMARIZE().append_hmmix_info(ibdmixafr, prefix + ".txt", prefixn, inference = "ibdmix", individualID = individualID[i])
        if unrec_label == "SUPER":
            prefix = prefix = summarypref + filepref + str(inds[i]) + ".nea.summary"
            prefixn = prefix + ".ibdmix"
            SUMMARIZE().append_hmmix_info(ibdmixafr, prefix + ".txt", prefixn, inference = "ibdmix", individualID = individualID[i])
#    os.remove(prefix + ".txt")


snpinfo = "/global/secure0/groups/mcb-aux01-access/zhangyulin9806/EVOCEANIA_singer/snpinfo/EVO_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask"
mutage = "/global/secure0/groups/mcb-aux01-access/zhangyulin9806/EVOCEANIA_singer/snpinfo/EVO_biall_snpinfo.human_ancestor.archaic.afr.outgroup.strictmask.mutage"
bcfpref = "/global/secure0/groups/mcb-aux01-access/zhangyulin9806/EVOCEANIA_singer/vcf_files/EVOCEANIA_subset_YRI_chr"


for i in range(len(inds)):
    k = df[df['ID']==inds[i]]['SuperGroup'].values[0]
    summarys = {
        "OCN":summarypref + filepref + str(inds[i]) + ".summary.hmmix.ibdmix.txt",
        "EUR":summarypref + filepref + str(inds[i]) + ".summary.hmmix.ibdmix.txt",
        "EAS":summarypref + filepref + str(inds[i]) + ".summary.hmmix.ibdmix.txt",
        "SEA":summarypref + filepref + str(inds[i]) + ".summary.hmmix.ibdmix.txt",
        "AFR":summarypref + filepref + str(inds[i]) + ".summary.ibdmix.txt",
        }
    summarys_nea = {
        "OCN":summarypref + filepref + str(inds[i]) + ".nea.summary.hmmix.ibdmix.txt",
        "EUR":summarypref + filepref + str(inds[i]) + ".nea.summary.hmmix.ibdmix.txt",
        "EAS":summarypref + filepref + str(inds[i]) + ".nea.summary.hmmix.ibdmix.txt",
        "SEA":summarypref + filepref + str(inds[i]) + ".nea.summary.hmmix.ibdmix.txt",
        "AFR":summarypref + filepref + str(inds[i]) + ".nea.summary.ibdmix.txt",
    }
    hap = "left" if inds[i]%2 == 0 else "right"
    SUMMARIZE().final_ind_count(
        samplename = individualID[i], 
        summary = summarys[k], 
        hap = hap, 
        snpinfo = snpinfo, 
        bcfpref = bcfpref, 
        outpref = summarypref + filepref + str(inds[i]) + ".count"
    )
    if unrec_label == "SUPER":
        SUMMARIZE().final_ind_count(
            samplename = individualID[i], 
            summary = summarys_nea[k], 
            hap = hap, 
            snpinfo = snpinfo, 
            bcfpref = bcfpref, 
            outpref = summarypref + filepref + str(inds[i]) + ".nea.count"
        )
    if unrec_label == "Ghost":
        pref = [summarypref + filepref + str(inds[i])]
    elif unrec_label == "SUPER":
        pref = [
            summarypref + filepref + str(inds[i]),
            summarypref + filepref + str(inds[i]) + ".nea",
        ]
    for j in range(len(pref)):
        SUMMARIZE().append_t1_t2(
            npzpref = summarypref + filepref + str(inds[i]) + f".{chrom}.xss",
            summary = pref[j] + ".count_DAF.txt", 
            snpinfo = mutage + f".{chrom}.txt", 
            func = "median",
            outpref = pref[j] + ".count_t1t2",
        )
        os.remove(pref[j] + ".count_DAF.txt")
        SUMMARIZE().merge_df_count_branch_mut(
            count = pref[j] + ".count.txt", 
            count_t1t2 = pref[j] + ".count_t1t2.txt", 
            output = pref[j] + ".count_merge.txt",
            pop = df[df['ID']==inds[i]]['SuperGroup'].values[0],
            unrec_label = unrec_label
        )
        os.remove(pref[j] + ".count.txt")
        os.remove(pref[j] + ".count_t1t2.txt")

if unrec_label == "Ghost":
    xssfile = summarypref + xss_15k_pref + str(inds[i]) + f".{chrom}.xss.npz"
    with np.load(xssfile) as d:
        data = {k: d[k] for k in d.files} 
    states = data["states"]
    treespan_phy = data["treespan_phy"]
    window_size = int(treespan_phy[0, 1] - treespan_phy[0, 0])
    nea_states = np.zeros(shape=states.shape, dtype=states.dtype)
    den_states = np.zeros(shape=states.shape, dtype=states.dtype)
    ghost_states = np.zeros(shape=states.shape, dtype=states.dtype)
    df = pd.read_csv(summarypref + filepref + str(inds[i]) + ".count_merge.txt", sep="\t")
    for j in range(len(df)):
        if df.loc[j, 'assign_label'] == "NEA":
            nea_states[int(df.loc[j, 'start'] / window_size):int(df.loc[j, 'end'] / window_size)] = 1
        elif df.loc[j, 'assign_label'] == "DEN":
            den_states[int(df.loc[j, 'start'] / window_size):int(df.loc[j, 'end'] / window_size)] = 1
        elif df.loc[j, 'assign_label'] == unrec_label:
            ghost_states[int(df.loc[j, 'start'] / window_size):int(df.loc[j, 'end'] / window_size)] = 1
    data["nea_states"] = nea_states
    data["den_states"] = den_states
    data["ghost_states"] = ghost_states
    np.savez_compressed(
        summarypref + xss_15k_pref + str(inds[i]) + f".{chrom}.xss.npz", **data
    )

